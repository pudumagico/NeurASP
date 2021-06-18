import sys
sys.path.append('../../')

import torch
import json

from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP

from translate import translate

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################
with open("CLEVR_val_questions.json") as fp:
    questions = json.load(fp)["questions"]

for q in questions:

    dprogram = r'''
    nn(label(1,I,B), ["obj(B,cylinder,large,gray,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,gray,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,gray,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,gray,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,gray,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,gray,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,blue,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,blue,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,blue,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,blue,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,blue,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,blue,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,brown,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,brown,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,brown,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,brown,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,brown,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,brown,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,yellow,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,yellow,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,yellow,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,yellow,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,yellow,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,yellow,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,red,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,red,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,red,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,red,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,red,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,red,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,green,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,green,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,green,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,green,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,green,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,green,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,purple,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,purple,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,purple,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,purple,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,purple,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,purple,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,large,cyan,metal,X1,Y1,X2,Y2)", "obj(B,sphere,large,cyan,metal,X1,Y1,X2,Y2)", "obj(B,cube,large,cyan,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,large,cyan,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,large,cyan,rubber,X1,Y1,X2,Y2)", "obj(B,cube,large,cyan,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,gray,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,gray,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,gray,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,gray,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,gray,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,gray,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,blue,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,blue,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,blue,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,blue,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,blue,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,blue,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,brown,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,brown,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,brown,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,brown,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,brown,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,brown,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,yellow,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,yellow,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,yellow,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,yellow,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,yellow,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,yellow,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,red,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,red,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,red,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,red,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,red,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,red,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,green,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,green,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,green,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,green,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,green,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,green,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,purple,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,purple,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,purple,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,purple,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,purple,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,purple,rubber,X1,Y1,X2,Y2)", "obj(B,cylinder,small,cyan,metal,X1,Y1,X2,Y2)", "obj(B,sphere,small,cyan,metal,X1,Y1,X2,Y2)", "obj(B,cube,small,cyan,metal,X1,Y1,X2,Y2)", "obj(B,cylinder,small,cyan,rubber,X1,Y1,X2,Y2)", "obj(B,sphere,small,cyan,rubber,X1,Y1,X2,Y2)", "obj(B,cube,small,cyan,rubber,X1,Y1,X2,Y2)", "other"]) :- box(I,B,X1,Y1,X2,Y2).
    '''

    aspProgram = translate(q["program"])

    with open("spatial_reasoning_enhanced.lp", "r") as fp:
         aspProgram += fp.read().replace("\n", " ")

    ########
    # Define nnMapping
    ########

    m = Net()
    nnMapping = {'label': m}

    ########
    # Construct a list of facts and a list of dataDic, where each dataDic maps terms to tensors
    ########

    # set the term and the path to the image files represetned by this term
    termPath = 'img ./data/'
    # set the size of the reshaped image
    img_size = 416
    # set the set of classes that we consider
    domain = ['LaGraMeCy', 'LaGraMeSp', 'LaGraMeCu', 'LaGraRuCy', 'LaGraRuSp', 'LaGraRuCu', 'LaBlMeCy', 'LaBlMeSp', 'LaBlMeCu', 'LaBlRuCy', 'LaBlRuSp', 'LaBlRuCu', 'LaBrMeCy', 'LaBrMeSp', 'LaBrMeCu', 'LaBrRuCy', 'LaBrRuSp', 'LaBrRuCu', 'LaYeMeCy', 'LaYeMeSp', 'LaYeMeCu', 'LaYeRuCy', 'LaYeRuSp', 'LaYeRuCu', 'LaReMeCy', 'LaReMeSp', 'LaReMeCu', 'LaReRuCy', 'LaReRuSp', 'LaReRuCu', 'LaGreMeCy', 'LaGreMeSp', 'LaGreMeCu', 'LaGreRuCy', 'LaGreRuSp', 'LaGreRuCu', 'LaPuMeCy', 'LaPuMeSp', 'LaPuMeCu', 'LaPuRuCy', 'LaPuRuSp', 'LaPuRuCu', 'LaCyMeCy', 'LaCyMeSp', 'LaCyMeCu', 'LaCyRuCy', 'LaCyRuSp', 'LaCyRuCu', 'SmGraMeCy', 'SmGraMeSp', 'SmGraMeCu', 'SmGraRuCy', 'SmGraRuSp', 'SmGraRuCu', 'SmBlMeCy', 'SmBlMeSp', 'SmBlMeCu', 'SmBlRuCy', 'SmBlRuSp', 'SmBlRuCu', 'SmBrMeCy', 'SmBrMeSp', 'SmBrMeCu', 'SmBrRuCy', 'SmBrRuSp', 'SmBrRuCu', 'SmYeMeCy', 'SmYeMeSp', 'SmYeMeCu', 'SmYeRuCy', 'SmYeRuSp', 'SmYeRuCu', 'SmReMeCy', 'SmReMeSp', 'SmReMeCu', 'SmReRuCy', 'SmReRuSp', 'SmReRuCu', 'SmGreMeCy', 'SmGreMeSp', 'SmGreMeCu', 'SmGreRuCy', 'SmGreRuSp', 'SmGreRuCu', 'SmPuMeCy', 'SmPuMeSp', 'SmPuMeCu', 'SmPuRuCy', 'SmPuRuSp', 'SmPuRuCu', 'SmCyMeCy', 'SmCyMeSp', 'SmCyMeCu', 'SmCyRuCy', 'SmCyRuSp', 'SmCyRuCu', "other"]

    factsList, dataList = termPath2dataList(termPath, img_size, domain)

    ########
    # Start inference for each image
    ########

    for idx, facts in enumerate(factsList):
        # Initialize NeurASP object
        NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
        # Find the most probable stable model
        models = NeurASPobj.infer(dataDic=dataList[idx], obs='', mvpp=aspProgram + facts)
        print('\nInfernece Result on Data {}:'.format(idx+1))
        print(models[0])
