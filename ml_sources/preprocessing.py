def pre_for_segmentation(image):
    # image = load_rgb(input_image)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    for i in range(mask.shape[2]):
        temp = skimage.io.imread(input_image)
        for j in range(temp.shape[2]):
            temp[:, :, j] = temp[:, :, j] * mask[:, :, i]
        plt.figure(figsize=(8, 8))

    global timestr
    timestr = time.strftime("%Y%m%d-%H%M%S")
    skimage.io.imsave('static/segmentation_img/' + str(timestr) + '.png', temp)