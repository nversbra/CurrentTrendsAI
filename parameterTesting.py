import csv
import os



parameters=open('parameters.txt', 'rb')
reader = csv.reader(parameters, delimiter=',')

result = open('result.txt', 'wb')
mywriter= csv.writer(result)

testdata=[]
test = open('test-file.csv', 'rb')
treader = csv.reader(test, delimiter=',')
for testd in treader:
    testdata.append(testd[0])
test.close()

for row in reader:
    print row
    command = 'Python DeepBlueNote.py training-data-file-0.csv test-data-file-0.csv output-file.csv ' + row[0] + ' ' + \
              row[1] + ' ' + row[2] + ' ' + row[3] + ' ' + row[4] + ' ' + row[5] + ' ' + row[6]
    os.system(command)

    pred = open('output-file.csv', 'rb')
    preader = csv.reader(pred, delimiter=',')
    j = 0
    accuracy = 0
    for prediction in preader:
        if testdata[j] != prediction[0]:
            accuracy += 1
        j+= 1

    pred.close()


    mywriter.writerow([accuracy, row[0], row[1], row[2]])


result.close()