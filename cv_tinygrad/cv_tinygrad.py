from tinygrad import Tensor, dtypes

def to_grayscale(image):
    gray = Tensor([0.299, 0.587, 0.114])
    return Tensor.einsum("abc,c->ab", image, gray) / 255

def convolve(image, f):
    image = image.unsqueeze(0).unsqueeze(0)
    f = f.unsqueeze(0).unsqueeze(0)
    image = image.conv2d(f, stride=1, padding=1)
    return image.squeeze(0).squeeze(0)

def sobel_filter(image):
    """
    Apply Sobel filter to the input image. This provides a good approximation of the gradient.
    """
    image = image.unsqueeze(0).unsqueeze(0)
    Kx = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
    Ky = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
    Ix = image.conv2d(Kx, padding=1)
    Iy = image.conv2d(Ky, padding=1)
    return Ix.squeeze(0).squeeze(0), Iy.squeeze(0).squeeze(0)


def horn_schunck(image1, image2, num_iter=10, alpha=0.4):
    """
        Horn Schunck method for computing dense optical flo
    """
    horizontal_flow = Tensor.zeros_like(image1)
    vertical_flow = Tensor.zeros_like(image2)

    # kernel_x = Tensor([[-1, 1], [-1, 1]]) * 0.25
    # kernel_y = Tensor([[-1, -1], [1, 1]]) * 0.25
    # kernel_t = Tensor([[1, 1], [1, 1]]) * 0.25
    kernel_laplacian = Tensor(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]]
    )

    fx1, fy1 = sobel_filter(image1)
    fx2, fy2 = sobel_filter(image2)

    fx = fx1 + fx2
    fy = fy1 + fy2
    ft = image2 - image1
    # ft = -fx1 + fx2 - fy1 + fy2
    # fx = convolve(image1, kernel_x) + convolve(image2, kernel_x)
    # fy = convolve(image1, kernel_y) + convolve(image2, kernel_y)
    # ft = convolve(image1, -kernel_t) + convolve(image2, kernel_t)

    for _ in range(num_iter):
        horizontal_flow_avg = convolve(horizontal_flow, kernel_laplacian)
        vertical_flow_avg = convolve(vertical_flow, kernel_laplacian)

        p = fx * horizontal_flow_avg + fy * vertical_flow_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2

        horizontal_flow = horizontal_flow_avg - fx * (p / d)
        vertical_flow = vertical_flow_avg - fy * (p / d)

    return horizontal_flow, -vertical_flow
