# -*- coding: utf-8 -*-

from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, diff, reformat_input, reformat_input_batched, get_paragraph, merge_to_free, download_and_unzip, calculate_md5, download_and_unzip_all
from .config import *
from .detection import get_textbox
from bidi import get_display
import sys
import json
import shutil
from logging import getLogger

if sys.version_info[0] == 2:
    from io import open
    from pathlib2 import Path
else:
    from pathlib import Path

LOGGER = getLogger(__name__)


class Reader(object):

    def __init__(self, lang_list, recognizer=True):

        self.device = 'cpu'
        self.recognizer = recognizer

        Path(MODULE_PATH).mkdir(parents=True, exist_ok=True)

        corrupt_msg = 'MD5 hash mismatch, possible file corruption'

        self.character_storage_directory = os.path.join(MODULE_PATH, 'character')
        if os.path.exists(self.character_storage_directory) == False:
            LOGGER.warning('Downloading character text files, please wait. '
                            'This may take several minutes depending upon your network connection.')
            download_and_unzip_all(character_folder['url'], MODULE_PATH, True)
            LOGGER.info('Download complete.')
        else: # check if all files are intact
            sums = []
            for filename in os.listdir(self.character_storage_directory):
                if filename.endswith(".txt"):
                    sums.append(calculate_md5(os.path.join(self.character_storage_directory, filename)))
            if sums != character_md5sum:
                shutil.rmtree(self.character_storage_directory)
                LOGGER.warning('Re-downloading character text files, please wait. '
                                'This may take several minutes depending upon your network connection.')
                download_and_unzip_all(character_folder['url'], MODULE_PATH, True)
                LOGGER.info('Download complete.')

        self.recognition_storage_directory = os.path.join(MODULE_PATH, 'model')
        Path(self.recognition_storage_directory).mkdir(parents=True, exist_ok=True)
        
        separator_list = {}

        unknown_lang = set(lang_list) - set(all_lang_list)
        if unknown_lang != set():
            raise ValueError(''.join(unknown_lang) + ' is not supported')
        # choose recognition model
        if lang_list == ['en']:
            self.setModelLanguage('english', lang_list, ['en'], '["en"]')
            recognition_model = recognition_models['gen2']['english_g2']
            # detection_model   = detection_models['gen2']['english_g2']
        elif 'th' in lang_list:
            self.setModelLanguage('thai', lang_list, ['th','en'], '["th","en"]')
            recognition_model = recognition_models['gen1']['thai_g1']
            # detection_model   = detection_models['gen1']['thai_g1']
        elif 'ch_tra' in lang_list:
            self.setModelLanguage('chinese_tra', lang_list, ['ch_tra','en'], '["ch_tra","en"]')
            recognition_model = recognition_models['gen1']['zh_tra_g1']
            # detection_model   = detection_models['gen1']['zh_tra_g1']
        elif 'ch_sim' in lang_list:
            self.setModelLanguage('chinese_sim', lang_list, ['ch_sim','en'], '["ch_sim","en"]')
            recognition_model = recognition_models['gen2']['zh_sim_g2']
            # detection_model   = detection_models['gen2']['zh_sim_g2']
        elif 'ja' in lang_list:
            self.setModelLanguage('japanese', lang_list, ['ja','en'], '["ja","en"]')
            recognition_model = recognition_models['gen2']['japanese_g2']
            # detection_model   = detection_models['gen2']['japanese_g2']
        elif 'ko' in lang_list:
            self.setModelLanguage('korean', lang_list, ['ko','en'], '["ko","en"]')
            recognition_model = recognition_models['gen2']['korean_g2']
            # detection_model   = detection_models['gen2']['korean_g2']
        elif 'ta' in lang_list:
            self.setModelLanguage('tamil', lang_list, ['ta','en'], '["ta","en"]')
            recognition_model = recognition_models['gen1']['tamil_g1']
            # detection_model   = detection_models['gen1']['tamil_g1']
        elif 'te' in lang_list:
            self.setModelLanguage('telugu', lang_list, ['te','en'], '["te","en"]')
            recognition_model = recognition_models['gen2']['telugu_g2']
            # detection_model   = detection_models['gen2']['telugu_g2']
        elif 'kn' in lang_list:
            self.setModelLanguage('kannada', lang_list, ['kn','en'], '["kn","en"]')
            recognition_model = recognition_models['gen2']['kannada_g2']
            # detection_model   = detection_models['gen2']['kannada_g2']
        elif set(lang_list) & set(bengali_lang_list):
            self.setModelLanguage('bengali', lang_list, bengali_lang_list+['en'], '["bn","as","en"]')
            recognition_model = recognition_models['gen1']['bengali_g1']
            # detection_model   = detection_models['gen1']['bengali_g1']
        elif set(lang_list) & set(arabic_lang_list):
            self.setModelLanguage('arabic', lang_list, arabic_lang_list+['en'], '["ar","fa","ur","ug","en"]')
            recognition_model = recognition_models['gen1']['arabic_g1']
            # detection_model   = detection_models['gen1']['arabic_g1']
        elif set(lang_list) & set(devanagari_lang_list):
            self.setModelLanguage('devanagari', lang_list, devanagari_lang_list+['en'], '["hi","mr","ne","en"]')
            recognition_model = recognition_models['gen1']['devanagari_g1']
            # detection_model   = detection_models['gen1']['devanagari_g1']
        elif set(lang_list) & set(cyrillic_lang_list):
            self.setModelLanguage('cyrillic', lang_list, cyrillic_lang_list+['en'],
                                    '["ru","rs_cyrillic","be","bg","uk","mn","en"]')
            recognition_model = recognition_models['gen2']['cyrillic_g2']
            # detection_model   = detection_models['gen2']['cyrillic_g2']
        else:
            self.model_lang = 'latin'
            recognition_model = recognition_models['gen2']['latin_g2']
            # detection_model   = detection_models['gen2']['latin_g2']

        self.character = recognition_model['characters']

        self.detection_path = os.path.join(self.recognition_storage_directory, detection_model['filename'])
        if os.path.isfile(self.detection_path) == False:
            LOGGER.warning('Downloading detection model, please wait. '
                            'This may take several minutes depending upon your network connection.')
            download_and_unzip(detection_model['url'], detection_model['filename'], self.recognition_storage_directory, True)
            assert calculate_md5(self.detection_path) == detection_model['md5sum'], corrupt_msg
            LOGGER.info('Download complete.')
        elif calculate_md5(self.detection_path) != detection_model['md5sum']:
            os.remove(self.detection_path)
            LOGGER.warning('Re-downloading the detection model, please wait. '
                            'This may take several minutes depending upon your network connection.')
            download_and_unzip(detection_model['url'], detection_model['filename'], self.recognition_storage_directory, True)
            assert calculate_md5(self.detection_path) == detection_model['md5sum'], corrupt_msg
            LOGGER.info('Download complete')

        self.recognition_path = os.path.join(self.recognition_storage_directory, recognition_model['filename'])
        # check recognition model file
        if recognizer:
            if os.path.isfile(self.recognition_path) == False:
                LOGGER.warning('Downloading recognition model, please wait. '
                                'This may take several minutes depending upon your network connection.')
                download_and_unzip(recognition_model['url'], recognition_model['filename'], self.recognition_storage_directory, True)
                assert calculate_md5(self.recognition_path) == recognition_model['md5sum'], corrupt_msg
                LOGGER.info('Download complete.')
            elif calculate_md5(self.recognition_path) != recognition_model['md5sum']:
                os.remove(self.recognition_path)
                LOGGER.warning('Re-downloading the recognition model, please wait. '
                                'This may take several minutes depending upon your network connection.')
                download_and_unzip(recognition_model['url'], recognition_model['filename'], self.recognition_storage_directory, True)
                assert calculate_md5(self.recognition_path) == recognition_model['md5sum'], corrupt_msg
                LOGGER.info('Download complete')

        self.setLanguageList(lang_list, recognition_model)

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        if recognizer:
            self.converter = get_recognizer(self.character, separator_list, dict_list)

    
    def setModelLanguage(self, language, lang_list, list_lang, list_lang_string):
        self.model_lang = language
        if set(lang_list) - set(list_lang) != set():
            if language == 'ch_tra' or language == 'ch_sim':
                language = 'chinese'
            raise ValueError(language.capitalize() + ' is only compatible with English, try lang_list=' + list_lang_string)

    def getChar(self, fileName):
        char_file = os.path.join(self.character_storage_directory, fileName)
        with open(char_file, "r", encoding="utf-8-sig") as input_file:
            list = input_file.read().splitlines()
            char = ''.join(list)
        return char
    
    def setLanguageList(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(self.character_storage_directory, lang + "_char.txt")
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
                                    detection_path = self.detection_path,
                                    max_candidates = max_candidates
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

        if self.recognizer==False:
            LOGGER.warning("Recognition is disabled, to enable use .Reader(recognizer=True)")
            return

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
        if ((batch_size == 1) or (self.device == 'cpu')):
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.converter, image_list, self.recognition_path,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.converter, image_list, self.recognition_path,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              self.device)
                result += result0

        if self.model_lang == 'arabic':
            direction_mode = 'rtl'
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
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


    def readtextlang(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
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
       
        char = []
        directory = self.character_storage_directory
        for i in range(len(result)):
            char.append(result[i][1])
        
        def search(arr,x):
            g = False
            for i in range(len(arr)):
                if arr[i]==x:
                    g = True
                    return 1
            if g == False:
                return -1
        def tupleadd(i):
            a = result[i]
            b = a + (filename[0:2],)
            return b
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(self.character_storage_directory, filename),'rt',encoding="utf8") as myfile:  
                    chartrs = str(myfile.read().splitlines()).replace('\n','') 
                    for i in range(len(char)):
                        res = search(chartrs,char[i])
                        if res != -1:
                            if filename[0:2]=="en" or filename[0:2]=="ch":
                                print(tupleadd(i))
    

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