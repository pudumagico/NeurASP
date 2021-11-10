import json
import re
import sys

sys.path.append('../../')

from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP
from translate import func_to_asp
from tqdm import tqdm

with open("CLEVR_v1.0/questions/CLEVR_val_questions.json") as fp:
    questions = json.load(fp)["questions"]

dprogram = r'''
    nn(label(1,I,B), [obj(B,cylinder,large,gray,metal,X1,Y1,X2,Y2), obj(B,sphere,large,gray,metal,X1,Y1,X2,Y2), obj(B,cube,large,gray,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,gray,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,gray,rubber,X1,Y1,X2,Y2), obj(B,cube,large,gray,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,blue,metal,X1,Y1,X2,Y2), obj(B,sphere,large,blue,metal,X1,Y1,X2,Y2), obj(B,cube,large,blue,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,blue,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,blue,rubber,X1,Y1,X2,Y2), obj(B,cube,large,blue,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,brown,metal,X1,Y1,X2,Y2), obj(B,sphere,large,brown,metal,X1,Y1,X2,Y2), obj(B,cube,large,brown,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,brown,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,brown,rubber,X1,Y1,X2,Y2), obj(B,cube,large,brown,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,yellow,metal,X1,Y1,X2,Y2), obj(B,sphere,large,yellow,metal,X1,Y1,X2,Y2), obj(B,cube,large,yellow,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,cube,large,yellow,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,red,metal,X1,Y1,X2,Y2), obj(B,sphere,large,red,metal,X1,Y1,X2,Y2), obj(B,cube,large,red,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,red,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,red,rubber,X1,Y1,X2,Y2), obj(B,cube,large,red,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,green,metal,X1,Y1,X2,Y2), obj(B,sphere,large,green,metal,X1,Y1,X2,Y2), obj(B,cube,large,green,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,green,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,green,rubber,X1,Y1,X2,Y2), obj(B,cube,large,green,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,purple,metal,X1,Y1,X2,Y2), obj(B,sphere,large,purple,metal,X1,Y1,X2,Y2), obj(B,cube,large,purple,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,purple,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,purple,rubber,X1,Y1,X2,Y2), obj(B,cube,large,purple,rubber,X1,Y1,X2,Y2), obj(B,cylinder,large,cyan,metal,X1,Y1,X2,Y2), obj(B,sphere,large,cyan,metal,X1,Y1,X2,Y2), obj(B,cube,large,cyan,metal,X1,Y1,X2,Y2), obj(B,cylinder,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,sphere,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,cube,large,cyan,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,gray,metal,X1,Y1,X2,Y2), obj(B,sphere,small,gray,metal,X1,Y1,X2,Y2), obj(B,cube,small,gray,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,gray,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,gray,rubber,X1,Y1,X2,Y2), obj(B,cube,small,gray,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,blue,metal,X1,Y1,X2,Y2), obj(B,sphere,small,blue,metal,X1,Y1,X2,Y2), obj(B,cube,small,blue,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,blue,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,blue,rubber,X1,Y1,X2,Y2), obj(B,cube,small,blue,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,brown,metal,X1,Y1,X2,Y2), obj(B,sphere,small,brown,metal,X1,Y1,X2,Y2), obj(B,cube,small,brown,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,brown,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,brown,rubber,X1,Y1,X2,Y2), obj(B,cube,small,brown,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,yellow,metal,X1,Y1,X2,Y2), obj(B,sphere,small,yellow,metal,X1,Y1,X2,Y2), obj(B,cube,small,yellow,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,cube,small,yellow,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,red,metal,X1,Y1,X2,Y2), obj(B,sphere,small,red,metal,X1,Y1,X2,Y2), obj(B,cube,small,red,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,red,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,red,rubber,X1,Y1,X2,Y2), obj(B,cube,small,red,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,green,metal,X1,Y1,X2,Y2), obj(B,sphere,small,green,metal,X1,Y1,X2,Y2), obj(B,cube,small,green,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,green,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,green,rubber,X1,Y1,X2,Y2), obj(B,cube,small,green,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,purple,metal,X1,Y1,X2,Y2), obj(B,sphere,small,purple,metal,X1,Y1,X2,Y2), obj(B,cube,small,purple,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,purple,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,purple,rubber,X1,Y1,X2,Y2), obj(B,cube,small,purple,rubber,X1,Y1,X2,Y2), obj(B,cylinder,small,cyan,metal,X1,Y1,X2,Y2), obj(B,sphere,small,cyan,metal,X1,Y1,X2,Y2), obj(B,cube,small,cyan,metal,X1,Y1,X2,Y2), obj(B,cylinder,small,cyan,rubber,X1,Y1,X2,Y2), obj(B,sphere,small,cyan,rubber,X1,Y1,X2,Y2), obj(B,cube,small,cyan,rubber,X1,Y1,X2,Y2)]) :- box(I,B,X1,Y1,X2,Y2).
    '''

termPath = f'img ./CLEVR_v1.0/images/val'
img_size = 480
domain = ['LaGraMeCy', 'LaGraMeSp', 'LaGraMeCu', 'LaGraRuCy', 'LaGraRuSp', 'LaGraRuCu', 'LaBlMeCy', 'LaBlMeSp', 'LaBlMeCu', 'LaBlRuCy',
          'LaBlRuSp', 'LaBlRuCu', 'LaBrMeCy', 'LaBrMeSp', 'LaBrMeCu', 'LaBrRuCy', 'LaBrRuSp', 'LaBrRuCu', 'LaYeMeCy', 'LaYeMeSp',
          'LaYeMeCu', 'LaYeRuCy', 'LaYeRuSp', 'LaYeRuCu', 'LaReMeCy', 'LaReMeSp', 'LaReMeCu', 'LaReRuCy', 'LaReRuSp', 'LaReRuCu',
          'LaGreMeCy', 'LaGreMeSp', 'LaGreMeCu', 'LaGreRuCy', 'LaGreRuSp', 'LaGreRuCu', 'LaPuMeCy', 'LaPuMeSp', 'LaPuMeCu', 'LaPuRuCy',
          'LaPuRuSp', 'LaPuRuCu', 'LaCyMeCy', 'LaCyMeSp', 'LaCyMeCu', 'LaCyRuCy', 'LaCyRuSp', 'LaCyRuCu', 'SmGraMeCy', 'SmGraMeSp',
          'SmGraMeCu', 'SmGraRuCy', 'SmGraRuSp', 'SmGraRuCu', 'SmBlMeCy', 'SmBlMeSp', 'SmBlMeCu', 'SmBlRuCy', 'SmBlRuSp', 'SmBlRuCu',
          'SmBrMeCy', 'SmBrMeSp', 'SmBrMeCu', 'SmBrRuCy', 'SmBrRuSp', 'SmBrRuCu', 'SmYeMeCy', 'SmYeMeSp', 'SmYeMeCu', 'SmYeRuCy',
          'SmYeRuSp', 'SmYeRuCu', 'SmReMeCy', 'SmReMeSp', 'SmReMeCu', 'SmReRuCy', 'SmReRuSp', 'SmReRuCu', 'SmGreMeCy', 'SmGreMeSp',
          'SmGreMeCu', 'SmGreRuCy', 'SmGreRuSp', 'SmGreRuCu', 'SmPuMeCy', 'SmPuMeSp', 'SmPuMeCu', 'SmPuRuCy', 'SmPuRuSp', 'SmPuRuCu',
          'SmCyMeCy', 'SmCyMeSp', 'SmCyMeCu', 'SmCyRuCy', 'SmCyRuSp', 'SmCyRuCu']

factsList, dataList = termPath2dataList(termPath, img_size, domain)

correct = 0
incorrect = 0
invalid = 0
total = 0

questionCounter = 0

with open("theory.lp", "r") as fp:
    theory = fp.read()

for q in tqdm(questions):
    #if questionCounter < 86300:
    #    questionCounter += 1
    #    continue

    aspProgram = func_to_asp(q["program"])
    aspProgram += theory

    m = Net()
    nnMapping = {'label': m}

    facts = factsList[q['image_index']]
    NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
    models = NeurASPobj.infer(dataDic=dataList[q['image_index']], obs='', mvpp=aspProgram + facts)
    answer = [atom for atom in models[0] if re.search(r"ans\(.*\)", atom)]

    if len(answer) > 0:
        answer = answer[0][4:-1]
    else:
        answer = "invalid"

    if answer == "true":
        answer = "yes"
    elif answer == "false":
        answer = "no"
    else:
        answer = str(answer)

    if answer == str(q['answer']):
        correct += 1
    elif answer == "invalid":
        invalid += 1
    else:
        incorrect += 1

    total += 1

print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")
