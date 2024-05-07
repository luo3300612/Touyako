def clip_bbox(bbox, im_shape):
    """
    bbox: [xmin, ymin, xmax, ymax]
    """
    im_h = im_shape[0]
    im_w = im_shape[1]
    bbox[0] = min(im_w, max(0, bbox[0]))
    bbox[1] = min(im_h, max(0, bbox[1]))
    bbox[2] = min(im_w, max(0, bbox[2]))
    bbox[3] = min(im_h, max(0, bbox[3]))
    return bbox


def expand_bbox(bbox, mode="default", expand_w=0, expand_h=0,
                expand_left=0, expand_right=0, expand_top=0, expand_bottom=0):
    assert mode in ["default", "center_wh_expand", "online", "cross_wh_ratio_expand",
                    "customize", "four_sides_pixel_expand"]
    xmin, ymin, xmax, ymax = bbox
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    if mode in ["center_wh_expand", "default"]:
        # xmin -= w * expand_w/2
        # xmax += w * expand_w/2
        # ymin -= h * expand_h/2
        # ymax += h * expand_h/2
        xmin -= w * expand_w
        xmax += w * expand_w
        ymin -= h * expand_h
        ymax += h * expand_h
    elif mode in ["cross_wh_ratio_expand", "online"]:
        # xmin -= h * expand_w/2
        # xmax += h * expand_w/2
        # ymin -= w * expand_h/2
        # ymax += w * expand_h/2
        xmin -= h * expand_w
        xmax += h * expand_w
        ymin -= w * expand_h
        ymax += w * expand_h
    elif mode in ["four_sides_pixel_expand", "customize"]:
        xmin -= expand_left
        xmax += expand_right
        ymin -= expand_top
        ymax += expand_bottom
    else:
        raise ValueError(f"expand_bbox: Error Expand Mode:{mode}")
    return [int(xmin), int(ymin), int(xmax), int(ymax)]
