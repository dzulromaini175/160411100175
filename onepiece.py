from math import log10
import requests
from bs4 import BeautifulSoup
import sqlite3
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from math import log10


src = "https://thoughtcatalog.com/category/creepy/"

page = requests.get(src)


soup = BeautifulSoup(page.content, 'html.parser')

artikel = soup.findAll(class_='tcf-article-md-content')
   
koneksi = sqlite3.connect('One_piece.db')
koneksi.execute(''' CREATE TABLE if not exists Onepiece
            (judul TEXT NOT NULL,
             isi TEXT NOT NULL);''')

for i in range(len(artikel)):
    try:
        link = artikel[i].find('a')['href']
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        #soup = soup.encode("utf-8")
        judul = soup.find(class_='entry-title').getText()
        judul = judul.encode()
        isi = soup.find(class_='entry-block-group box-content')
        paragraf = isi.findAll('p')
        p = ''
        for s in paragraf:
            
            p+=str(s.getText().encode("utf-8"))[2:-1] +' '
            
        koneksi.execute('INSERT INTO Onepiece values (?,?)', (judul, p));
    except:
        pass

koneksi.commit()
tampil = koneksi.execute("SELECT * FROM Onepiece")

with open ('data_crawler.csv', newline='', mode='w', encoding = 'utf-8')as employee_file :
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for i in tampil:
        employee_writer.writerow(i)

tampil = koneksi.execute("SELECT * FROM Onepiece")
isi = []
for row in tampil:
    isi.append(row[1])
    #print(row)
    

print("crawl sudah")
#vsm
tmp = ''
for i in isi:
    tmp = tmp + ' ' +i


stop_words = set(stopwords.words('english')) 
ps = PorterStemmer()

word_tokens = word_tokenize(tmp) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 

katadasar=[]
for word in filtered_sentence:
    tmp = ps.stem(word)
    if not tmp in katadasar:
        katadasar.append(tmp)


matrix = []
for row in isi :
    tamp_isi=[]
    for a in katadasar:
        tamp_isi.append(row.lower().count(a))
    matrix.append(tamp_isi)

#print(katadasar)
#for m in matrix:
 #   print(m)
 #import csv kata yg sesuai dengan KBI

with open ('data_matrix.csv', newline='', mode='w', encoding = 'utf-8')as employee_file :
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(katadasar)
    for i in matrix :
        employee_writer.writerow(i)

#tf-idf
df = list()
for d in range (len(matrix[0])):
    total = 0
    for i in range(len(matrix)):
        if matrix[i][d] !=0:
            total += 1
    df.append(total)

idf = list()
for i in df:
    tmp = 1 + log10(len(matrix)/(1+i))
    idf.append(tmp)

tf = matrix
tfidf = []
for baris in range(len(matrix)):
    tampungBaris = []
    for kolom in range(len(matrix[0])):
        tmp = tf[baris][kolom] * idf[kolom]
        tampungBaris.append(tmp)
    tfidf.append(tampungBaris)


with open('tf-idf.csv', newline='', mode='w', encoding = 'utf-8') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(katadasar)
    for i in tfidf:
        employee_writer.writerow(i)


def pearsonCalculate(data, u,v):
    "i, j is an index"
    atas=0; bawah_kiri=0; bawah_kanan = 0
    for k in range(len(data)):
        atas += (data[k,u] - meanFitur[u]) * (data[k,v] - meanFitur[v])
        bawah_kiri += (data[k,u] - meanFitur[u])**2
        bawah_kanan += (data[k,v] - meanFitur[v])**2
    bawah_kiri = bawah_kiri ** 0.5
    bawah_kanan = bawah_kanan ** 0.5
    return atas/(bawah_kiri * bawah_kanan)
def meanF(data):
    meanFitur=[]
    for i in range(len(data[0])):
        meanFitur.append(sum(data[:,i])/len(data))
    return np.array(meanFitur)
def seleksiFiturPearson(katadasar, data, threshold):
    global meanFitur
    data = np.array(data)
    meanFitur = meanF(data)
    u=0
    while u < len(data[0]):
        dataBaru=data[:, :u+1]
        meanBaru=meanFitur[:u+1]
        katadasarBaru=katadasar[:u+1]
        v = u
        while v < len(data[0]):
            if u != v:
                value = pearsonCalculate(data, u,v)
                if value < threshold:
                    dataBaru = np.hstack((dataBaru, data[:, v].reshape(data.shape[0],1)))
                    meanBaru = np.hstack((meanBaru, meanFitur[v]))
                    katadasarBaru = np.hstack((katadasarBaru, katadasar[v]))
            v+=1
        data = dataBaru
        meanFitur=meanBaru
        katadasar=katadasarBaru
        if u%50 == 0 : print("proses : ", u, data.shape)
        u+=1
    return katadasar,data

katadasarBaru, fiturBaru = seleksiFiturPearson(katadasar, tfidf,0.783);
kmeans = KMeans(n_clusters=3, random_state=0).fit(tfidf);
print(kmeans.labels_)
classnya=kmeans.labels_
s_avg = silhouette_score(fiturBaru, classnya, random_state=0)
print (s_avg)
with open('Anggota_cluster.csv', newline='', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in classnya.reshape(-1,1):
        employee_writer.writerow(i)

with open('Seleksi_Fitur.csv', newline='', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([katadasarBaru.tolist()])
    for i in fiturBaru:
        employee_writer.writerow(i)
