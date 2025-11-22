from fastapi import APIRouter, UploadFile, HTTPException, Form, File, status
import numpy as np
import cv2
from app.face_engine import FaceEngine
from app.database import User, SessionLocal

router = APIRouter()
engine = FaceEngine()


# ======================================================
# IMAGE READER
# ======================================================
def read_image(file: UploadFile):
    data = file.file.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ======================================================
# REGISTER USER
# ======================================================
@router.post(
    "/register",
    summary="Register a user with password / face / both",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "password": {"type": "string"},
                            "login_method": {"type": "string"},
                            "front": {"type": "string", "format": "binary"},
                            "left": {"type": "string", "format": "binary"},
                            "right": {"type": "string", "format": "binary"},
                        },
                        "required": ["email", "login_method","front","left","right","password"],
                    }
                }
            }
        }
    }
)
async def register(
    email: str = Form(...),
    login_method: str = Form(...),
    password: str = Form(None),
    front: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None)
):

    if login_method not in ["password", "face", "both"]:
        raise HTTPException(400, "Invalid login method")

    if login_method in ["password", "both"] and not password:
        raise HTTPException(400, "Password required")

    embedding_bytes = None

    # ---------------- FACE LOGIN REQUIRED ----------------
    if login_method in ["face", "both"]:
        if not front or not left or not right:
            raise HTTPException(400, "3 face images required")

        img1, img2, img3 = read_image(front), read_image(left), read_image(right)
        r1, r2, r3 = engine.process(img1), engine.process(img2), engine.process(img3)

        if not r1 or not r2 or not r3:
            raise HTTPException(400, "Face not detected in all images")

        emb1 = np.frombuffer(r1["embedding"], dtype=np.float32)
        emb2 = np.frombuffer(r2["embedding"], dtype=np.float32)
        emb3 = np.frombuffer(r3["embedding"], dtype=np.float32)

        final = (emb1 + emb2 + emb3) / 3
        final = final / np.linalg.norm(final)

        embedding_bytes = final.astype(np.float32).tobytes()

    # ---------------- SAVE TO DATABASE ----------------
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(400, "Email already exists")

        user = User(
            email=email,
            password=password,
            login_method=login_method,
            embedding=embedding_bytes
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {"message": "User registered", "id": user.id}
    finally:
        db.close()


# ======================================================
# LOGIN (PASSWORD OR FACE)
# ======================================================
@router.post("/login")
async def login(
    login_type: str = Form(...),   # "password" | "face" | "both"
    email: str = Form(...),
    password: str = Form(None),
    image: UploadFile = File(None)
):
    if login_type not in ["password", "face", "both"]:
        raise HTTPException(400, "Invalid login type")

    db = SessionLocal()

    # Fetch user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(404, "User not found")

    # Check if login_type is allowed
    if login_type == "password" and user.login_method not in ["password", "both"]:
        raise HTTPException(403, "Password login disabled for this user")

    if login_type == "face" and user.login_method not in ["face", "both"]:
        raise HTTPException(403, "Face login disabled for this user")

    if login_type == "both" and user.login_method != "both":
        raise HTTPException(403, "This user must use only one method")

    # --------------------------------------------------
    # Password check (only if required)
    # --------------------------------------------------
    if login_type in ["password", "both"]:
        if not password:
            raise HTTPException(400, "Password required")
        if user.password != password:
            raise HTTPException(401, "Incorrect password")

    # --------------------------------------------------
    # Face check (only if required)
    # --------------------------------------------------
    if login_type in ["face", "both"]:
        if image is None:
            raise HTTPException(400, "Face image required")

        img = read_image(image)
        r = engine.process(img)
        if r is None:
            raise HTTPException(400, "Face not detected")

        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        stored = np.frombuffer(user.embedding, dtype=np.float32)

        score = float(np.dot(stored, emb))

        # threshold
        if score < 0.95:
            raise HTTPException(401, f"Face mismatch. Score={score}")

    return {
        "message": f"{login_type} login successful",
        "email": user.email,
        "used_method": login_type
    }


# ======================================================
# GET USERS LIST
# ======================================================
@router.get("/users")
def get_users():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [
            {
                "id": u.id,
                "email": u.email,
                "login_method": u.login_method
            }
            for u in users
        ]
    finally:
        db.close()


# ======================================================
# DELETE USER
# ======================================================
@router.delete("/user/delete")
async def delete_user(email: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()

        if not user:
            raise HTTPException(404, "User not found")

        if user.password != password:
            raise HTTPException(401, "Incorrect password")

        db.delete(user)
        db.commit()
        return {"message": "User deleted"}

    finally:
        db.close()


# ======================================================
# UPDATE PASSWORD
# ======================================================
@router.put("/user/password")
async def update_password(
    email: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form(...)
):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()

        if not u:
            raise HTTPException(404, "User not found")

        if u.password != current_password:
            raise HTTPException(401, "Incorrect current password")

        u.password = new_password
        db.commit()
        return {"message": "Password updated"}
    finally:
        db.close()


# ======================================================
# UPDATE FACE EMBEDDING
# ======================================================
@router.put("/user/face")
async def update_face(
    email: str = Form(...),
    password: str = Form(...),
    image: UploadFile = File(...)
):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()

        if not u:
            raise HTTPException(404, "User not found")

        if u.password != password:
            raise HTTPException(401, "Incorrect password")

        img = read_image(image)
        r = engine.process(img)
        if r is None:
            raise HTTPException(400, "Face not detected")

        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        u.embedding = emb.astype(np.float32).tobytes()
        db.commit()

        return {"message": "Face embedding updated"}

    finally:
        db.close()


# ======================================================
# UPDATE LOGIN METHOD (AFTER LOGIN)
# ======================================================
@router.put("/user/login-method", summary="Update login method after login")
async def update_login_method(
    email: str = Form(...),
    password: str = Form(...),
    new_method: str = Form(...)
):
    if new_method not in ["password", "face", "both"]:
        raise HTTPException(status_code=400, detail="Invalid login method")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.password != password:
            raise HTTPException(status_code=401, detail="Incorrect password")

        # update method
        user.login_method = new_method
        db.commit()
        db.refresh(user)

        return {
            "message": "Login method updated",
            "email": user.email,
            "new_method": new_method
        }

    finally:
        db.close()

