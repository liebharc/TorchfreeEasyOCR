# -*- coding: utf-8 -*-

from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, diff, reformat_input, reformat_input_batched, get_paragraph, merge_to_free
from .config import BASE_PATH, recognition_models, imgH
from .detection import get_textbox
import os
import json



class Reader(object):

    def __init__(self, recognizer=True):

        self.device = 'cpu'
        
        separator_list = {}

        self.model_lang = 'english'
        model = recognition_models['gen2']['english_g2']
        self.character = model['characters']

        self.setLanguageList(['en'], model)

        dict_list = {}
        for lang in ['en']:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        if recognizer:
            self.converter = get_recognizer(self.character, separator_list, dict_list)

    
    def setModelLanguage(self, language):
        self.model_lang = language

    def setLanguageList(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                char_list =  input_file.read().splitlines()
            self.lang_char += char_list
        if model.get('symbols'):
            symbol = model['symbols']
        elif model.get('character_list'):
            symbol = model['character_list']
        else:
            symbol = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        self.lang_char = set(self.lang_char).union(set(symbol))
        self.lang_char = ''.join(self.lang_char)

    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None,
               threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
               ):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box_list = get_textbox(img, 
                                    canvas_size = canvas_size, 
                                    mag_ratio = mag_ratio,
                                    text_threshold = text_threshold, 
                                    link_threshold = link_threshold, 
                                    low_text = low_text,
                                    poly = False, 
                                    device = self.device, 
                                    optimal_num_chars = optimal_num_chars,
                                    threshold = threshold, 
                                    bbox_min_score = bbox_min_score, 
                                    bbox_min_size = bbox_min_size, 
                                    max_candidates = max_candidates,
                                    )

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  rotation_info = None,paragraph = False,\
                  contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                  y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard'):

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))


        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # without gpu/parallelization, it is faster to process image one by one
        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              self.device)
                result += result0


        direction_mode = 'ltr'

        if paragraph:
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode = direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            if paragraph:
                return [ {'boxes':item[0],'text':item[1]} for item in result]    
            return [ {'boxes':item[0],'text':item[1],'confident':item[2]} for item in result]
        elif output_format == 'json':
            if paragraph:
                return [json.dumps({'boxes':[list(map(int, lst)) for lst in item[0]],'text':item[1]}, ensure_ascii=False) for item in result]
            return [json.dumps({'boxes':[list(map(int, lst)) for lst in item[0]],'text':item[1],'confident':item[2]}, ensure_ascii=False) for item in result]
        elif output_format == 'free_merge':
            return merge_to_free(result, free_list)
        else:
            return result

    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                 threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                 output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail, rotation_info,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, y_ths, x_ths, False, output_format)

        return result
    
    def readtext_batched(self, image, n_width=None, n_height=None,\
                        decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                        workers = 0, allowlist = None, blocklist = None, detail = 1,\
                        rotation_info = None, paragraph = False, min_size = 20,\
                        contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                        text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                        canvas_size = 2560, mag_ratio = 1.,\
                        slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                        width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                        threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                        output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        When sending a list of images, they all must of the same size,
        the following parameters will automatically resize if they are not None
        n_width: int, new width
        n_height: int, new height
        '''
        img, img_cv_grey = reformat_input_batched(image, n_width, n_height)

        horizontal_list_agg, free_list_agg = self.detect(img, 
                                                    min_size = min_size, text_threshold = text_threshold,\
                                                    low_text = low_text, link_threshold = link_threshold,\
                                                    canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                    slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                    height_ths = height_ths, width_ths= width_ths,\
                                                    add_margin = add_margin, reformat = False,\
                                                    threshold = threshold, bbox_min_score = bbox_min_score,\
                                                    bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                    )
        result_agg = []
        # put img_cv_grey in a list if its a single img
        img_cv_grey = [img_cv_grey] if len(img_cv_grey.shape) == 2 else img_cv_grey
        for grey_img, horizontal_list, free_list in zip(img_cv_grey, horizontal_list_agg, free_list_agg):
            result_agg.append(self.recognize(grey_img, horizontal_list, free_list,\
                                            decoder, beamWidth, batch_size,\
                                            workers, allowlist, blocklist, detail, rotation_info,\
                                            paragraph, contrast_ths, adjust_contrast,\
                                            filter_ths, y_ths, x_ths, False, output_format))

        return result_agg