from .Neural_style_transfer import nst
import os

def main(style,content,id):

    s = str(style)
    c = str(content)

    # print(s,c)

    # print(os.getcwd(),"ok",s)

    media_path = 'media/'
    s = media_path + s
    c = media_path + c

    final_path = nst(s,c,id)

    return final_path