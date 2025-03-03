from PIL import Image
import numpy as np
from onnxruntime import InferenceSession
from .utils import CTCLabelConverter
import math


def np_softmax(x, dim=-1):
    means = np.max(x, axis=dim, keepdims=True)
    x_exp = np.exp(x - means)
    x_exp_sum = np.sum(x_exp, axis=dim, keepdims=True)
    
    return x_exp / x_exp_sum

def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


class NumPyDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, drop_last=False):
        """
        A custom NumPy-based DataLoader replacement for torch.utils.data.DataLoader.

        Args:
            dataset: The dataset object (should support `__getitem__` and `__len__`).
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset before iterating.
            num_workers (int): Placeholder, only `0` is supported (no multiprocessing).
            collate_fn (callable, optional): Function to merge samples into a batch.
            drop_last (bool): If True, drops the last batch if it's smaller than `batch_size`.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers  # Not used
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.indices = np.arange(len(dataset))
        self.reset()

    def reset(self):
        """Shuffle indices if required and reset batch index."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_index = 0

    def __iter__(self):
        """Returns an iterator."""
        self.reset()
        return self

    def __next__(self):
        """Fetch the next batch of data."""
        if self.batch_index >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.batch_index:self.batch_index + self.batch_size]
        self.batch_index += self.batch_size

        batch = [self.dataset[i] for i in batch_indices]
        
        if self.collate_fn:
            batch = self.collate_fn(batch)

        return batch

    def __len__(self):
        """Number of batches per epoch."""
        total_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            total_batches += 1
        return total_batches


class NormalizePAD:
    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size  # (C, H, W)
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img = (img - 0.5) / 0.5  # Scale between -1 and 1

        if img.ndim == 2:  # If grayscale (H, W), add channel dim
            img = np.expand_dims(img, axis=0)  # Now shape is (1, H, W)

        c, h, w = self.max_size
        Pad_img = np.zeros((c, h, w), dtype=np.float32)  # Zero-filled array of max_size

        # Ensure input fits in the padding space
        img_c, img_h, img_w = img.shape
        Pad_img[:, :, :img_w] = img  # Right pad

        if img_w < w:  # Add border padding
            Pad_img[:, :, img_w:] = np.expand_dims(img[:, :, img_w - 1], axis=2).repeat(w - img_w, axis=2)

        return Pad_img


class ListDataset:
    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, 'L')


class AlignCollate:
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast=0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size

            # Augmentation: Adjust contrast if needed
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            ratio = w / float(h)
            resized_w = self.imgW if math.ceil(self.imgH * ratio) > self.imgW else math.ceil(self.imgH * ratio)
            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            
            resized_images.append(transform(resized_image))

        image_tensors = np.stack([np.expand_dims(t, axis=0) for t in resized_images], axis=0)  # NumPy equivalent of torch.cat()
        return image_tensors


def recognizer_predict(converter, test_loader, batch_max_length, recognition_path,\
                       ignore_idx, char_group_idx, decoder = 'greedy', beamWidth= 5, device = 'cpu'):
    result = []

    for image_tensors in test_loader:
        batch_size = image_tensors.shape[0]
        image = image_tensors[0]

        ort_session = InferenceSession(recognition_path)
        ort_inputs = {ort_session.get_inputs()[0].name: image}
        ort_outs = ort_session.run(None, ort_inputs)
        preds = ort_outs[0]

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = np.array([preds.shape[1]] * batch_size, dtype=np.int32)

        ######## filter ignore_char, rebalance
        preds_prob = np_softmax(preds, dim=2)

        preds_prob[:,:,ignore_idx] = 0.
        pred_norm = preds_prob.sum(axis=2)
        preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
        preds_prob = np.asarray(preds_prob, dtype=np.float32)

        if decoder == 'greedy':
            preds_index = np.argmax(preds_prob, axis=2)
            preds_index = preds_index.reshape(-1)

            preds_str = converter.decode_greedy(preds_index, preds_size)
        elif decoder == 'beamsearch':
            preds_str = converter.decode_beamsearch(preds_prob, beamWidth=beamWidth)
        elif decoder == 'wordbeamsearch':
            preds_str = converter.decode_wordbeamsearch(preds_prob, beamWidth=beamWidth)
        else:
            raise ValueError(f"No such decoder: {decoder}")

        values = preds_prob.max(axis=2)
        indices = preds_prob.argmax(axis=2)
        preds_max_prob = []
        for v,i in zip(values, indices):
            max_probs = v[i!=0]
            if len(max_probs)>0:
                preds_max_prob.append(max_probs)
            else:
                preds_max_prob.append(np.array([0]))

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            confidence_score = custom_mean(pred_max_prob)
            result.append([pred, confidence_score])

    return result
    

def get_recognizer(character, separator_list, dict_list):

    converter = CTCLabelConverter(character, separator_list, dict_list)

    return converter

def get_text(character, imgH, imgW, converter, image_list, recognition_path,\
             ignore_char = '', decoder = 'greedy', beamWidth =5, batch_size=1, contrast_ths=0.1,\
             adjust_contrast=0.5, filter_ths = 0.003, device = 'cpu'):
    batch_max_length = int(imgW/10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try: ignore_idx.append(character.index(char)+1)
        except: pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = NumPyDataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=AlignCollate_normal, drop_last=False)

    # predict first round
    result1 = recognizer_predict(converter, test_loader,batch_max_length, recognition_path,\
                                 ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    # predict second round
    low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = NumPyDataLoader(
                        test_data, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=AlignCollate_contrast, drop_last=False)
        result2 = recognizer_predict(converter, test_loader, batch_max_length, recognition_path,\
                                     ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1]>pred2[1]:
                result.append( (box, pred1[0], pred1[1]) )
            else:
                result.append( (box, pred2[0], pred2[1]) )
        else:
            result.append( (box, pred1[0], pred1[1]) )

    return result
