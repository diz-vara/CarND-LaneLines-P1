imshape = image.shape
imCenterX = imshape[1] / 2
imCenterY = imshape[0] / 2
maskWidthTop = imshape[1] * 0.2
maskWidthBottom = imshape[1] * 1.1
maskTop = imCenterY - imshape[0] * 0.1
maskBottom = imshape[0]
vertices = np.array([[(imCenterX-maskWidthTop/2,maskTop),(imCenterX + maskWidthTop/2, maskTop), 
(imCenterX + maskWidthBottom/2, maskBottom), (imCenterX-maskWidthBottom/2,maskBottom)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
