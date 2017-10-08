#-*- coding:utf-8 -*-
import re
import math

def getwords(doc):
    splitter = re.compile('\\W*') # 알파벳이나 숫자를 제외한 문자
    #텍스트를 알파벳이 아닌 문자로 분리
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]
    
    #유일한 단어들만 리턴
    return dict([(w,1) for w in words])
    
class classifier:
    def __init__(self, getfeatures, filename=None):
        # 특성/분류 조합수를 셈
        self.fc={}
        # 각 분류별 문서 수를 셈
        self.cc={}
        self.getfeatures = getfeatures
    
    # 특성/분류 쌍 횟수를 증가시킴
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1
        
    # 분류 횟수를 증가시킴
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1
        
    # 한 특성이 특정 분류에 출현한 횟수
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0
        
    # 분류당 항목 개수
    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0
        
    # 항목 전체 개수
    def totalcount(self):
        return sum(self.cc.values())
        
    # 전체 분류 목록
    def categories(self):
        return self.cc.keys()
        
    def train(self, item, cat):
        features=self.getfeatures(item)
        #이 분류 내 모든 특성 횟수를 증가
        for f in features:
            self.incf(f, cat)
        
        #이 분류 횟수를 증가시킴
        self.incc(cat)
    
    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        
        #해당 분류에서 특성이 나타난 횟수를 그 분류에 있는 전체 항목 개수로 나눔
        return self.fcount(f, cat)/self.catcount(cat)
        
    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        #현재의 확률을 계산함
        basicprob=prf(f, cat)
        
        #모든 분류에 이 특성이 출현한 횟수를 계산함
        totals=sum([self.fcount(f,c) for c in self.categories() ])
        
        #가중편균을 계산함
        bp = ((weight*ap + (totals*basicprob)) / (weight+totals))
        return bp
        
class naivebayes(classifier):
    def docprob(self, item, cat):
        features=self.getfeatures(item)
        
        #모든 특성의 확률을 곱함
        p=1
        for f in features:
            p*=self.weightedprob(f, cat, self.fprob)
        return p
    
    def prob(self, item, cat):
        catprob=self.catcount(cat)/self.totalcount()
        docprob=self.docprob(item, cat)
        return docprob*catprob
        
def sampletrain(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick brown fox jumps', 'good')
    