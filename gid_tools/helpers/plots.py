import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter


def display_samples_with_feedback(sample_images, sample_labels, feedback_vector):
    fig, axes = plt.subplots(4, 5, figsize=(12, 5))
    for ax, img, lbl, area in zip(axes.flatten(), sample_images, sample_labels, feedback_vector):
        # convert to H×W numpy in [0,1] for display
        im = img.squeeze().cpu().numpy()
        im_disp = (im + 1) / 2
        ax.imshow(im_disp, cmap='gray')
        ax.set_title(f"Label: {lbl}\nArea: {area}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def render_T_image(lengths, thicknesses, angles, phi,
                   canvas_size=28, upscale=10, blur_radius=1.2):
    """
    Render a T-shape from 4 brackets given sampler parameters.
    """
    L1, L2, L3, L4 = lengths
    T1, T2, T3, T4 = thicknesses
    theta2, theta3, theta4 = angles

    # High-resolution canvas
    HR = canvas_size * upscale
    im = Image.new("L", (HR, HR), 0)
    draw = ImageDraw.Draw(im)

    # Compute unrotated points
    O  = (0.0, 0.0)
    P1 = (L3, 0.0)
    P2 = (P1[0] + L4 * np.cos(theta4), P1[1] + L4 * np.sin(theta4))
    Q1 = (P1[0] + L1 * np.cos(theta3), P1[1] + L1 * np.sin(theta3))
    Q2 = (Q1[0] + L2 * np.cos(theta3 + theta2),
          Q1[1] + L2 * np.sin(theta3 + theta2))

    # Rotate by phi
    c, s = np.cos(phi), np.sin(phi)
    def rot(pt):
        x, y = pt
        return (x * c - y * s, x * s + y * c)
    pts = [rot(p) for p in (O, P1, P2, Q1, Q2)]

    # Scale and center
    xs, ys = zip(*pts)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = (HR * 0.8) / max(maxx - minx, maxy - miny)
    shift = HR * 0.1
    pix = [((x - minx) * scale + shift, (y - miny) * scale + shift) for x, y in pts]
    O_p, P1_p, P2_p, Q1_p, Q2_p = pix

    # Draw lines
    draw.line([O_p,  P1_p], fill=255, width=int(T3 * scale))
    draw.line([P1_p, P2_p], fill=255, width=int(T4 * scale))
    draw.line([P1_p, Q1_p], fill=255, width=int(T1 * scale))
    draw.line([Q1_p, Q2_p], fill=255, width=int(T2 * scale))

    # Blur and downsample
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return im.resize((canvas_size, canvas_size), Image.LANCZOS)


def render_V_image(lengths, thicknesses, angles, phi,
                             canvas_size=28, upscale=10, blur_radius=1.2):
    """
    Render a branching V-shape from 4 brackets given sampler parameters.
    """
    L1, L2, L3, L4 = lengths
    T1, T2, T3, T4 = thicknesses
    theta2, theta3, theta4 = angles

    # High-resolution canvas
    HR = canvas_size * upscale
    im = Image.new("L", (HR, HR), 0)
    draw = ImageDraw.Draw(im)

    # Compute unrotated points
    J  = (0.0, 0.0)
    A1 = (L1, 0.0)
    A2 = (A1[0] + L2 * np.cos(theta2), A1[1] + L2 * np.sin(theta2))
    B1 = (L3 * np.cos(theta3), L3 * np.sin(theta3))
    B2 = (B1[0] + L4 * np.cos(theta3 + theta4),
          B1[1] + L4 * np.sin(theta3 + theta4))

    # Rotate by phi
    c, s = np.cos(phi), np.sin(phi)
    pts = [(x * c - y * s, x * s + y * c) for x, y in (J, A1, A2, B1, B2)]

    # Scale and center
    xs, ys = zip(*pts)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = (HR * 0.8) / max(maxx - minx, maxy - miny)
    shift = HR * 0.1
    pix = [((x - minx) * scale + shift, (y - miny) * scale + shift) for x, y in pts]
    J_p, A1_p, A2_p, B1_p, B2_p = pix

    # Draw lines
    draw.line([J_p,  A1_p], fill=255, width=int(T1 * scale))
    draw.line([A1_p, A2_p], fill=255, width=int(T2 * scale))
    draw.line([J_p,  B1_p], fill=255, width=int(T3 * scale))
    draw.line([B1_p, B2_p], fill=255, width=int(T4 * scale))

    # Blur and downsample
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return im.resize((canvas_size, canvas_size), Image.LANCZOS)


def make_bracket_image(lengths, thicknesses, angles, global_rot,
                       canvas_size=28, upscale=10, blur_radius=1.2,
                       fill=False):
    HR = canvas_size * upscale
    im = Image.new("L", (HR, HR), 0)
    draw = ImageDraw.Draw(im)
    # build points
    pts = [(0,0)]
    ang = 0.0
    for length, theta in zip(lengths, (*angles, 0)):
        ang += theta
        x,y = pts[-1]
        pts.append((x + length*np.cos(ang), y + length*np.sin(ang)))
    # global rotation
    c,s = np.cos(global_rot), np.sin(global_rot)
    pts = [(x*c - y*s, x*s + y*c) for x,y in pts]
    # scale & center
    xs, ys = zip(*pts)
    w, h = max(xs)-min(xs), max(ys)-min(ys)
    scale = (HR*0.8)/max(w,h)
    m = HR*0.1
    pts = [((x-min(xs))*scale + m, (y-min(ys))*scale + m) for x,y in pts]
    # draw
    if fill:
        draw.polygon(pts, fill=255)
    else:
        for i in range(len(pts)-1):
            draw.line([pts[i], pts[i+1]],
                      fill=255,
                      width=int(thicknesses[i]*scale))
    # blur + downsample
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return im.resize((canvas_size,canvas_size), Image.LANCZOS)