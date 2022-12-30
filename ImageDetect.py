import cv2
import os
import sys
from loguru import logger

SAVE_PATH = r'Result'
ICON_FOLDER = r'Icon'

METRO_ICON = {
    'Calculator': 'metro_calculator.png',
    'Chrome': 'metro_chrome.png',
    'CMD': 'metro_cmd.png',
    'Cortana': 'metro_cortana.png',
    'Edge': 'metro_edge.png',
    'IE': 'metro_ie.png',
    'Office': 'metro_office.png',
    'Paint': 'metro_paint.png',
    'Powershell': 'metro_powershell.png'
    }

TASKBAR_ICON = {
    'Calculator': 'taskbar_calculator.png',
    'CMD': 'taskbar_cmd.png',
    'Chrome': 'taskbar_chrome.png',
    'Edge': 'taskbar_edge.png',
    'Powershell': 'taskbar_powershell.png',
    'Powershell ISE': 'taskbar_powershell_ise.png',
    'Snipping Tool': 'taskbar_snipping_tool.png',
    'Sticky Notes': 'taskbar_sticky_notes.png'
    }

METRO1 = {
    'Screenshot': r'Screenshot\Metro1.png',
    'Icon': METRO_ICON
    }

METRO2 = {
    'Screenshot': r'Screenshot\Metro2.png',
    'Icon': METRO_ICON
    }

TASKBAR1 = {
    'Screenshot': r'Screenshot\Taskbar1.png',
    'Icon': TASKBAR_ICON
    }

TASKBAR2 = {
    'Screenshot': r'Screenshot\Taskbar2.png',
    'Icon': TASKBAR_ICON
    }

IMG_SET_LIST = [
    METRO1, METRO2, TASKBAR1, TASKBAR2
    ]


class Image:
    def __init__(self, img_set_list: list):
        self.img_set_list = img_set_list

    def main(self):
        for name, img in map(self.detect, self.img_set_list):
            cv2.imwrite(os.path.join(SAVE_PATH, name), img)

    def detect(self, img_set: dict) -> tuple[cv2.Mat, str]:
        img_screenshot_filename = os.path.basename(img_set['Screenshot'])
        logger.info('Loading images...')
        img_screenshot = self.read(img_set['Screenshot'])
        icon = {key: self.read(os.path.join(ICON_FOLDER, value)) for key, value in img_set['Icon'].items()}
        result = {}
        logger.info('Matching start!')
        for img_icon_name, img_icon in icon.items():
            max_loc, max_val = self.matchLocVal(img_screenshot, img_icon)
            if max_val > 0.9:
                logger.debug(f'[similarity = {max_val:.3f}, location = {max_loc}] : {img_icon_name}')
                logger.debug(f'type:{type(img_screenshot)}')
                """
                Hightlight the matched image, uncomment the line to use it.
                cv2.putText   : Put the icon name on the upper left corner of the matched image.
                cv2.circle    : Put a dot on the center of the matched image.
                cv2.rectangle : Put a rectangle to surround the matched image.
                """
                # cv2.putText(img_screenshot, img_icon_name, (max_loc[0], max_loc[1]), 
                #             cv2.FONT_HERSHEY_COMPLEX, fontScale=0.35, color=(0,0,255), thickness=1)
                # cv2.circle(img_screenshot, (max_loc[0]+(img_icon.shape[0])//2, max_loc[1]+(img_icon.shape[1])//2),
                #            radius=6, color=(255,0,0), thickness=-1)
                # cv2.rectangle(img_screenshot, max_loc, (max_loc[0]+img_icon.shape[0], max_loc[1]+img_icon.shape[1]),
                #               color=(255,0,0), thickness=2)

                result[img_icon_name] = max_loc
            else:
                logger.warning(f'[similarity is under 0.9] : {img_icon_name}')
        logger.debug(f'Match : {result}')

        # Sort by location
        order_result = dict(sorted(result.items(), key=lambda item:(item[1][1], item[1][0])))
        logger.debug(f'Order : {list(order_result.keys())}')
        logger.info('Matching finished!')

        return (img_screenshot_filename, img_screenshot)

    def matchLocVal(self, img: cv2.Mat, img_templ: cv2.Mat, method=cv2.TM_CCOEFF_NORMED) -> tuple[tuple[int, int], float]:
        img_match = cv2.matchTemplate(img, img_templ, method)
        min_val, max_val, min_loc, max_loc =  cv2.minMaxLoc(img_match)
        return max_loc, max_val

    def read(self, path:str) -> cv2.Mat:
        # Check image path
        if not(os.path.isfile(path)):
            logger.error(f'Image file not exist : {path}')
            sys.exit()
        logger.success(f'Image file found : {path}')

        # Check image format
        img = cv2.imread(path)
        if img is None:
            logger.error(f'Image file load failed : {path}')
            sys.exit()
        logger.success(f'Image file load successfully : {path}')

        return img


if __name__ == '__main__':
    img = Image(IMG_SET_LIST)
    img.main()