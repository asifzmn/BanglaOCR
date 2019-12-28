import matplotlib.pyplot as plt
import numpy as np
import operator
import csv

def CSVRead():

    wordMeaning = {}
    with open('Dictionary/BengaliDictionary0.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)

        for row in reader:
            meanings = row[1:]
            while '' in meanings:meanings.remove('')
            wordMeaning[row[0]]=meanings
            # print(row[0],wordMeaning[row[0]])

    csvFile.close()
    return wordMeaning

def FrequencyMapping():

    fileNumber = 4

    letterFreq = {}
    letterWithKarFreq = {}

    for typeIteration in range(fileNumber):
        file = open("Dictionary/BengaliWordList"+str(typeIteration)+".txt", "r")
        wordList = file.read().split('\n')



        for word in wordList:
        # for word in ['সূক্ষ্ম']:

            # if "্" not in word:
            #     continue

            for letter in word:
                letterFreq[letter] = letterFreq.setdefault(letter, 0) + 1

            # print(word)

            indices = [index for index in range(len(word)) if word[index] == "্"]

            doubleComposite = []
            removeIndices = []

            for i in range(len(indices)-1) :
                if (abs(indices[i+1]-indices[i])<3):
                    removeIndices.append(indices[i+1])
                    doubleComposite.append(indices[i])

            indices = sorted(set(indices) - set(removeIndices))
            tempList = []

            for index in indices:
                tempList.append(index-1)

                karInclusion =  1 if index+2<len(word) and word[index+2] in kar else 0
                doubleCompound = 1 if index in doubleComposite else 0
                # doubleCompound = 0

                tempList.append(index+2+karInclusion+doubleCompound)

            tempList.insert(0,0)
            indices =tempList

            parts = [word[i:j] for i, j in zip(indices, indices[1:] + [None])]
            # print(parts)

            letterWithKarList = []
            for part in parts:
                if part=='':continue
                if '্' in part:
                    letterWithKarList.append(part)
                    continue
                for i in range(len(part)):
                    if part[i] in kar:continue
                    karInclusion = 1 if i + 1 < len(part) and part[i + 1] in kar else 0
                    letterWithKarList.append(part[i:i+1+karInclusion])

            # print(letterWithKarList)

            # for i in range(len(word)):
            #     if(not i == len(word)-1 and word[i+1])

            for letterWithKar in letterWithKarList:
                letterWithKarFreq[letterWithKar] = letterWithKarFreq.setdefault(letterWithKar, 0) + 1


    # sortedLetterFreq = sorted(letterFreq.items(), key=operator.itemgetter(1))
    # sortedLetterWithKarFreq = sorted(letterWithKarFreq.items(), key=operator.itemgetter(1))

    letterWithKarFreq = {key: val for key, val in letterWithKarFreq.items() if val > 1000}

    print(letterWithKarFreq)
    # for key,value in sortedLetterWithKarFreq.items():
    #     print(key,value)

    # plt.bar(letterFreq.keys(), letterFreq.values(), 1, color='g')
    # plt.bar(letterWithKarFreq.keys(), letterWithKarFreq.values(), 1, color='b')
    # plt.show()

    return letterFreq,letterWithKarFreq

def BasicsCombination():

    file = open("Dictionary/BengaliCharacterCombinations.txt", "r")
    basicCharacters = file.readline().split(' ')
    vowels = basicCharacters[:11]
    consonants = basicCharacters[11:]
    file.readline().split(' ')
    kar = file.readline().split(' ')
    fola = file.readline().split(' ')

    # print(basicCharacters)
    # print(vowels)
    # print(consonants)
    # print(kar)
    # print(fola)

    return basicCharacters,vowels,consonants,kar,fola

if __name__ == '__main__':

    # wordMeaning = CSVRead()
    # basicCharacters,vowels,consonants,kar,fola = BasicsCombination()
    # letterFreq,letterWithKarFreq, = FrequencyMapping()

    exit()