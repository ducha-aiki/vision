import torch
import torchvision.transforms as transforms
import unittest
import random
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

try:
    from scipy import stats
except ImportError:
    stats = None


GRACE_HOPPER = 'assets/grace_hopper_517x606.jpg'


class Tester(unittest.TestCase):

    def test_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2

        img = torch.ones(3, height, width)
        oh1 = (height - oheight) // 2
        ow1 = (width - owidth) // 2
        imgnarrow = img[:, oh1:oh1 + oheight, ow1:ow1 + owidth]
        imgnarrow.fill_(0)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.sum() == 0, "height: " + str(height) + " width: " \
                                  + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum1 = result.sum()
        assert sum1 > 1, "height: " + str(height) + " width: " \
                         + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum2 = result.sum()
        assert sum2 > 0, "height: " + str(height) + " width: " \
                         + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        assert sum2 > sum1, "height: " + str(height) + " width: " \
                            + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)

    def test_five_crop(self):
        to_pil_image = transforms.ToPILImage()
        h = random.randint(5, 25)
        w = random.randint(5, 25)
        for single_dim in [True, False]:
            crop_h = random.randint(1, h)
            crop_w = random.randint(1, w)
            if single_dim:
                crop_h = min(crop_h, crop_w)
                crop_w = crop_h
                transform = transforms.FiveCrop(crop_h)
            else:
                transform = transforms.FiveCrop((crop_h, crop_w))

            img = torch.FloatTensor(3, h, w).uniform_()
            results = transform(to_pil_image(img))

            assert len(results) == 5
            for crop in results:
                assert crop.size == (crop_w, crop_h)

            to_pil_image = transforms.ToPILImage()
            tl = to_pil_image(img[:, 0:crop_h, 0:crop_w])
            tr = to_pil_image(img[:, 0:crop_h, w - crop_w:])
            bl = to_pil_image(img[:, h - crop_h:, 0:crop_w])
            br = to_pil_image(img[:, h - crop_h:, w - crop_w:])
            center = transforms.CenterCrop((crop_h, crop_w))(to_pil_image(img))
            expected_output = (tl, tr, bl, br, center)
            assert results == expected_output

    def test_ten_crop(self):
        to_pil_image = transforms.ToPILImage()
        h = random.randint(5, 25)
        w = random.randint(5, 25)
        for should_vflip in [True, False]:
            for single_dim in [True, False]:
                crop_h = random.randint(1, h)
                crop_w = random.randint(1, w)
                if single_dim:
                    crop_h = min(crop_h, crop_w)
                    crop_w = crop_h
                    transform = transforms.TenCrop(crop_h,
                                                   vertical_flip=should_vflip)
                    five_crop = transforms.FiveCrop(crop_h)
                else:
                    transform = transforms.TenCrop((crop_h, crop_w),
                                                   vertical_flip=should_vflip)
                    five_crop = transforms.FiveCrop((crop_h, crop_w))

                img = to_pil_image(torch.FloatTensor(3, h, w).uniform_())
                results = transform(img)
                expected_output = five_crop(img)
                if should_vflip:
                    vflipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    expected_output += five_crop(vflipped_img)
                else:
                    hflipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    expected_output += five_crop(hflipped_img)

                assert len(results) == 10
                assert expected_output == results

    def test_resize(self):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        osize = random.randint(5, 12) * 2

        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(osize),
            transforms.ToTensor(),
        ])(img)
        assert osize in result.size()
        if height < width:
            assert result.size(1) <= result.size(2)
        elif width < height:
            assert result.size(1) >= result.size(2)

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([osize, osize]),
            transforms.ToTensor(),
        ])(img)
        assert osize in result.size()
        assert result.size(1) == osize
        assert result.size(2) == osize

        oheight = random.randint(5, 12) * 2
        owidth = random.randint(5, 12) * 2
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([oheight, owidth]),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

    def test_random_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        padding = random.randint(1, 20)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((oheight, owidth), padding=padding),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor()
        ])(img)
        assert result.size(1) == height
        assert result.size(2) == width
        assert np.allclose(img.numpy(), result.numpy())

    def test_pad(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = torch.ones(3, height, width)
        padding = random.randint(1, 20)
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(padding),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == height + 2 * padding
        assert result.size(2) == width + 2 * padding

    def test_pad_with_tuple_of_pad_values(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = transforms.ToPILImage()(torch.ones(3, height, width))

        padding = tuple([random.randint(1, 20) for _ in range(2)])
        output = transforms.Pad(padding)(img)
        assert output.size == (width + padding[0] * 2, height + padding[1] * 2)

        padding = tuple([random.randint(1, 20) for _ in range(4)])
        output = transforms.Pad(padding)(img)
        assert output.size[0] == width + padding[0] + padding[2]
        assert output.size[1] == height + padding[1] + padding[3]

    def test_pad_raises_with_invalid_pad_sequence_len(self):
        with self.assertRaises(ValueError):
            transforms.Pad(())

        with self.assertRaises(ValueError):
            transforms.Pad((1, 2, 3))

        with self.assertRaises(ValueError):
            transforms.Pad((1, 2, 3, 4, 5))

    def test_lambda(self):
        trans = transforms.Lambda(lambda x: x.add(10))
        x = torch.randn(10)
        y = trans(x)
        assert (y.equal(torch.add(x, 10)))

        trans = transforms.Lambda(lambda x: x.add_(10))
        x = torch.randn(10)
        y = trans(x)
        assert (y.equal(x))

    def test_to_tensor(self):
        channels = 3
        height, width = 4, 4
        trans = transforms.ToTensor()
        input_data = torch.ByteTensor(channels, height, width).random_(0, 255).float().div_(255)
        img = transforms.ToPILImage()(input_data)
        output = trans(img)
        assert np.allclose(input_data.numpy(), output.numpy())

        ndarray = np.random.randint(low=0, high=255, size=(height, width, channels))
        output = trans(ndarray)
        expected_output = ndarray.transpose((2, 0, 1)) / 255.0
        assert np.allclose(output.numpy(), expected_output)

    @unittest.skipIf(accimage is None, 'accimage not available')
    def test_accimage_to_tensor(self):
        trans = transforms.ToTensor()

        expected_output = trans(Image.open(GRACE_HOPPER).convert('RGB'))
        output = trans(accimage.Image(GRACE_HOPPER))

        self.assertEqual(expected_output.size(), output.size())
        assert np.allclose(output.numpy(), expected_output.numpy())

    @unittest.skipIf(accimage is None, 'accimage not available')
    def test_accimage_resize(self):
        trans = transforms.Compose([
            transforms.Resize(256, interpolation=Image.LINEAR),
            transforms.ToTensor(),
        ])

        expected_output = trans(Image.open(GRACE_HOPPER).convert('RGB'))
        output = trans(accimage.Image(GRACE_HOPPER))

        self.assertEqual(expected_output.size(), output.size())
        self.assertLess(np.abs((expected_output - output).mean()), 1e-3)
        self.assertLess((expected_output - output).var(), 1e-5)
        # note the high absolute tolerance
        assert np.allclose(output.numpy(), expected_output.numpy(), atol=5e-2)

    @unittest.skipIf(accimage is None, 'accimage not available')
    def test_accimage_crop(self):
        trans = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        expected_output = trans(Image.open(GRACE_HOPPER).convert('RGB'))
        output = trans(accimage.Image(GRACE_HOPPER))

        self.assertEqual(expected_output.size(), output.size())
        assert np.allclose(output.numpy(), expected_output.numpy())

    def test_tensor_to_pil_image(self):
        trans = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_data = torch.Tensor(3, 4, 4).uniform_()
        img = trans(img_data)
        assert img.getbands() == ('R', 'G', 'B')
        r, g, b = img.split()

        expected_output = img_data.mul(255).int().float().div(255)
        assert np.allclose(expected_output[0].numpy(), to_tensor(r).numpy())
        assert np.allclose(expected_output[1].numpy(), to_tensor(g).numpy())
        assert np.allclose(expected_output[2].numpy(), to_tensor(b).numpy())

        # single channel image
        img_data = torch.Tensor(1, 4, 4).uniform_()
        img = trans(img_data)
        assert img.getbands() == ('L',)
        l, = img.split()
        expected_output = img_data.mul(255).int().float().div(255)
        assert np.allclose(expected_output[0].numpy(), to_tensor(l).numpy())

    def test_tensor_gray_to_pil_image(self):
        trans = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_data_byte = torch.ByteTensor(1, 4, 4).random_(0, 255)
        img_data_short = torch.ShortTensor(1, 4, 4).random_()
        img_data_int = torch.IntTensor(1, 4, 4).random_()

        img_byte = trans(img_data_byte)
        img_short = trans(img_data_short)
        img_int = trans(img_data_int)
        assert img_byte.mode == 'L'
        assert img_short.mode == 'I;16'
        assert img_int.mode == 'I'

        assert np.allclose(img_data_short.numpy(), to_tensor(img_short).numpy())
        assert np.allclose(img_data_int.numpy(), to_tensor(img_int).numpy())

    def test_tensor_rgba_to_pil_image(self):
        trans = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_data = torch.Tensor(4, 4, 4).uniform_()
        img = trans(img_data)
        assert img.mode == 'RGBA'
        assert img.getbands() == ('R', 'G', 'B', 'A')
        r, g, b, a = img.split()

        expected_output = img_data.mul(255).int().float().div(255)
        assert np.allclose(expected_output[0].numpy(), to_tensor(r).numpy())
        assert np.allclose(expected_output[1].numpy(), to_tensor(g).numpy())
        assert np.allclose(expected_output[2].numpy(), to_tensor(b).numpy())
        assert np.allclose(expected_output[3].numpy(), to_tensor(a).numpy())

    def test_ndarray_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.ByteTensor(4, 4, 3).random_(0, 255).numpy()
        img = trans(img_data)
        assert img.getbands() == ('R', 'G', 'B')
        r, g, b = img.split()

        assert np.allclose(r, img_data[:, :, 0])
        assert np.allclose(g, img_data[:, :, 1])
        assert np.allclose(b, img_data[:, :, 2])

        # single channel image
        img_data = torch.ByteTensor(4, 4, 1).random_(0, 255).numpy()
        img = trans(img_data)
        assert img.getbands() == ('L',)
        l, = img.split()
        assert np.allclose(l, img_data[:, :, 0])

    def test_ndarray_bad_types_to_pil_image(self):
        trans = transforms.ToPILImage()
        with self.assertRaises(AssertionError):
            trans(np.ones([4, 4, 1], np.int64))
            trans(np.ones([4, 4, 1], np.uint16))
            trans(np.ones([4, 4, 1], np.uint32))
            trans(np.ones([4, 4, 1], np.float64))

    def test_ndarray_gray_float32_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.FloatTensor(4, 4, 1).random_().numpy()
        img = trans(img_data)
        assert img.mode == 'F'
        assert np.allclose(img, img_data[:, :, 0])

    def test_ndarray_gray_int16_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.ShortTensor(4, 4, 1).random_().numpy()
        img = trans(img_data)
        assert img.mode == 'I;16'
        assert np.allclose(img, img_data[:, :, 0])

    def test_ndarray_gray_int32_to_pil_image(self):
        trans = transforms.ToPILImage()
        img_data = torch.IntTensor(4, 4, 1).random_().numpy()
        img = trans(img_data)
        assert img.mode == 'I'
        assert np.allclose(img, img_data[:, :, 0])

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_vertical_flip(self):
        random_state = random.getstate()
        random.seed(42)
        img = transforms.ToPILImage()(torch.rand(3, 10, 10))
        vimg = img.transpose(Image.FLIP_TOP_BOTTOM)

        num_samples = 250
        num_vertical = 0
        for _ in range(num_samples):
            out = transforms.RandomVerticalFlip()(img)
            if out == vimg:
                num_vertical += 1

        p_value = stats.binom_test(num_vertical, num_samples, p=0.5)
        random.setstate(random_state)
        assert p_value > 0.0001

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_horizontal_flip(self):
        random_state = random.getstate()
        random.seed(42)
        img = transforms.ToPILImage()(torch.rand(3, 10, 10))
        himg = img.transpose(Image.FLIP_LEFT_RIGHT)

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transforms.RandomHorizontalFlip()(img)
            if out == himg:
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.5)
        random.setstate(random_state)
        assert p_value > 0.0001

    @unittest.skipIf(stats is None, 'scipt.stats is not available')
    def test_normalize(self):
        def samples_from_standard_normal(tensor):
            p_value = stats.kstest(list(tensor.view(-1)), 'norm', args=(0, 1)).pvalue
            return p_value > 0.0001

        random_state = random.getstate()
        random.seed(42)
        for channels in [1, 3]:
            img = torch.rand(channels, 10, 10)
            mean = [img[c].mean() for c in range(channels)]
            std = [img[c].std() for c in range(channels)]
            normalized = transforms.Normalize(mean, std)(img)
            assert samples_from_standard_normal(normalized)
        random.setstate(random_state)

    def test_adjust_brightness(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transforms.adjust_brightness(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = transforms.adjust_brightness(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [0, 2, 6, 27, 67, 113, 18, 4, 117, 45, 127, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = transforms.adjust_brightness(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 10, 26, 108, 255, 255, 74, 16, 255, 180, 255, 2]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_contrast(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transforms.adjust_contrast(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = transforms.adjust_contrast(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [43, 45, 49, 70, 110, 156, 61, 47, 160, 88, 170, 43]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = transforms.adjust_contrast(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 0, 0, 22, 184, 255, 0, 0, 255, 94, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_saturation(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transforms.adjust_saturation(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = transforms.adjust_saturation(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [2, 4, 8, 87, 128, 173, 39, 25, 138, 133, 215, 88]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = transforms.adjust_saturation(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 6, 22, 0, 149, 255, 32, 0, 255, 4, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_hue(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        with self.assertRaises(ValueError):
            transforms.adjust_hue(x_pil, -0.7)
            transforms.adjust_hue(x_pil, 1)

        # test 0: almost same as x_data but not exact.
        # probably because hsv <-> rgb floating point ops
        y_pil = transforms.adjust_hue(x_pil, 0)
        y_np = np.array(y_pil)
        y_ans = [0, 5, 13, 54, 139, 226, 35, 8, 234, 91, 255, 1]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 1
        y_pil = transforms.adjust_hue(x_pil, 0.25)
        y_np = np.array(y_pil)
        y_ans = [13, 0, 12, 224, 54, 226, 234, 8, 99, 1, 222, 255]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = transforms.adjust_hue(x_pil, -0.25)
        y_np = np.array(y_pil)
        y_ans = [0, 13, 2, 54, 226, 58, 8, 234, 152, 255, 43, 1]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjust_gamma(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transforms.adjust_gamma(x_pil, 1)
        y_np = np.array(y_pil)
        assert np.allclose(y_np, x_np)

        # test 1
        y_pil = transforms.adjust_gamma(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [0, 35, 57, 117, 185, 240, 97, 45, 244, 151, 255, 15]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

        # test 2
        y_pil = transforms.adjust_gamma(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 0, 0, 11, 71, 200, 5, 0, 214, 31, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        assert np.allclose(y_np, y_ans)

    def test_adjusts_L_mode(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_rgb = Image.fromarray(x_np, mode='RGB')

        x_l = x_rgb.convert('L')
        assert transforms.adjust_brightness(x_l, 2).mode == 'L'
        assert transforms.adjust_saturation(x_l, 2).mode == 'L'
        assert transforms.adjust_contrast(x_l, 2).mode == 'L'
        assert transforms.adjust_hue(x_l, 0.4).mode == 'L'
        assert transforms.adjust_gamma(x_l, 0.5).mode == 'L'

    def test_color_jitter(self):
        color_jitter = transforms.ColorJitter(2, 2, 2, 0.1)

        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')

        for i in range(10):
            y_pil = color_jitter(x_pil)
            assert y_pil.mode == x_pil.mode

            y_pil_2 = color_jitter(x_pil_2)
            assert y_pil_2.mode == x_pil_2.mode

    def test_linear_transformation(self):
        x = torch.randn(250, 10, 10, 3)
        flat_x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        # compute principal components
        sigma = torch.mm(flat_x.t(), flat_x) / flat_x.size(0)
        u, s, _ = np.linalg.svd(sigma.numpy())
        zca_epsilon = 1e-10  # avoid division by 0
        d = torch.Tensor(np.diag(1. / np.sqrt(s + zca_epsilon)))
        u = torch.Tensor(u)
        principal_components = torch.mm(torch.mm(u, d), u.t())
        # initialize whitening matrix
        whitening = transforms.LinearTransformation(principal_components)
        # pass first vector
        xwhite = whitening(x[0].view(10, 10, 3))
        # estimate covariance
        xwhite = xwhite.view(1, 300).numpy()
        cov = np.dot(xwhite, xwhite.T) / x.size(0)
        assert np.allclose(cov, np.identity(1), rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
