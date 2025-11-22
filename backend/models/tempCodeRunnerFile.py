    def get_centers(h, w, stride, num_anchors=2):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        centers = centers * stride
        centers = np.repeat(centers, num_anchors, axis=0)
        return centers

    centers_8  = get_centers(80, 80, 8)
    centers_16 = get_centers(40, 40, 16)
    centers_32 = get_centers(20, 20, 32)