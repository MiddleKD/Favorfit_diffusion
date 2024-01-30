class CLIPImagePreprocessor:
    def __init__(self, 
                 crop_size=[224,224],
                 do_center_crop=True,
                 do_convert_rgb=True,
                 do_normalize=True,
                 do_rescale=True,
                 do_resize=True,
                 image_mean=[0.48145466, 0.4578275, 0.40821073],
                 image_std=[0.26862954, 0.26130258, 0.27577711],
                 ):
        from transformers import CLIPImageProcessor

        self.clip_image_processor = CLIPImageProcessor(
            crop_size=crop_size,
            do_center_crop=do_center_crop,
            do_convert_rgb=do_convert_rgb,
            do_normalize=do_normalize,
            do_rescale=do_rescale,
            do_resize=do_resize,
            image_mean=image_mean,
            image_std=image_std,
            resample=3
        )

    def __call__(self, prompt_image):
        pixel_values = self.clip_image_processor(
            prompt_image,
            return_tensors="pt",
        ).pixel_values

        return pixel_values


class CLIPImageEncoder:
    def __init__(self, from_pretrained=False):
        from transformers import CLIPVisionConfig, CLIPVisionModel

        self.image_preprocessor = CLIPImagePreprocessor()
        if from_pretrained==True:
            self.clip_image_encoder = CLIPVisionModel.from_pretrained( "openai/clip-vit-base-patch32", cache_dir="./data") 
        else:
            self.clip_image_encoder = CLIPVisionModel(CLIPVisionConfig())

    def __call__(self, x):
        x = self.clip_image_encoder(x).last_hidden_state
        return x

if __name__ == "__main__":
    from PIL import Image

    image_preprocessor = CLIPImagePreprocessor()
    image_encoder = CLIPImageEncoder()

    from PIL import Image
    img_pil = Image.open("./test.jpg")
    preprocessed_img = image_preprocessor(img_pil)
    encoded_img = image_encoder(preprocessed_img)
    print(encoded_img.shape)
