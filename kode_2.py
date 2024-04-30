#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:25:59 2024

@author: williamnehemia
"""

import os
import re
import math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#from kode_2 import VSM



class VSM:
    def __init__(self, path, tf_doc_type, dft_type, normalization_type):
        self.path = path # menyimpan path dokumen
        self.document_num = len(os.listdir(path)) # menyimpan banyak dokumen
        self.list_of_words = [] # menyimpan daftar semua kata dari dokumen + query
        self.df_document = pd.DataFrame() # dataframe untuk menyimpan tf, dft, idf, tf idf dari dokumen dan query
        self.tf_doc_type = tf_doc_type # menyimpan tipe tf dokumen (natural, logarithm, augmented, boolean, log_ave)
        self.dft_type = dft_type # menyimpan tipe dft dokumen (no, idf, prob_idf)
        self.normalization_type = normalization_type # menyimpan tipe normalization dokumen (none, cosine)

    
    def set_query(self, query, tf_query_type, dft_query_type, normalization_query_type):
        self.query = query # menyimpan query
        self.tf_query_type = tf_query_type # menyimpan tipe tf query (natural, logarithm, augmented, boolean, log_ave)
        self.dft_query_type = dft_query_type # menyimpan tipe dft query (no, idf, prob_idf)
        self.normalization_query_type = normalization_query_type # menyimpan tipe normalization query (none, cosine)
    
    def generate_document_index(self):
        # mendapatkan  semua kata / term dari dokumen
        self.get_all_words()
        
        # generate tf document
        self.generate_tf_document(self.tf_doc_type)
        
        # generate tf - idf document
        self.generate_tf_idf_document(self.dft_type, self.normalization_type)
    
    def generate_query_index(self):
        # get all words from query
        # generate tf query
        self.generate_tf_query(self.tf_query_type)
        # generate tf idf query
        self.generate_tf_idf_query(self.dft_query_type, self.normalization_query_type)
    
    def get_all_words(self):
        curr_file = os.listdir(self.path) # mendapatkan list file pada directory
        for file_name in curr_file: # looping setiap dokumen
            with open(self.path + '/' + file_name, "r") as file:
                content = file.read() # baca konten pada dokumen
                
                content = content.lower() # mengubah huruf menjadi huruf kecil
                content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                tokens = content.split(" ")
                
                # looping per token dalam dokumen
                for token in tokens:
                    if token not in self.list_of_words:
                        self.list_of_words.append(token)
                        
        query_curr = self.query # mengambil query
        query_curr = query_curr.lower() # mengubah query menjadi huruf kecil
        query_curr = re.sub(r'[^a-zA-Z\s]', '', query_curr) # hanya mengabil huruf latin dari query
        tokens = query_curr.split(" ") # melakukan tokenisasi dari query
        for token in tokens: # looping per token
            if token not in self.list_of_words: # jika token dari query tidak ada dalam daftar kata
                self.list_of_words.append(token) # menambahkan token ke daftar kata
            
        
        self.df_document['word'] = self.list_of_words
    
    def generate_tf_document(self, tf_type): # tf_type dapat diganti dengan logarithm, augmented, boolean, dan log ave
        dft_temp = {key: 0 for key in self.list_of_words} # membuat dictionary untuk menyimpan jumlah kemunculan dokumen ada di berapa dokumen (dft)
        if tf_type == 'natural': # jika menggunakan tf natural
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = []
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ")
                    for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan jumlah kata sekarang pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df_document = pd.concat([self.df_document, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df_document['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
                        
                
        elif tf_type == 'logarithm':
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = []
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ")
                    for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        
                        if num_of_word == 0: # jika num_of_word adalah 0 maka tidak akan dihitung nilai log nya
                            num_of_word = 1
                        else: # num_of_word bukan 0
                            num_of_word = 1 + math.log10(num_of_word) # menghitung tf menggunakan logarithm
                        
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df_document = pd.concat([self.df_document, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df_document['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
        elif tf_type == 'augmented':
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = []
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ")
                    for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                maximum_num_of_words = max(list_of_num_of_words_temp) # menghitung nilai maximum tf
                list_of_num_of_words_temp = [(0.5 + ( (0.5 * x)  / maximum_num_of_words ) ) for x in list_of_num_of_words_temp] # menghitung tf dengan augmented
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df_document = pd.concat([self.df_document, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df_document['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
        elif tf_type == 'boolean':
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = []
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ")
                    for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                list_of_num_of_words_temp = [1 if x != 0 else 0 for x in list_of_num_of_words_temp] # menghitung tf dengan boolean
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df_document = pd.concat([self.df_document, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df_document['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
        else: # log ave
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = []
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ")
                    for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                avg_tf = sum(list_of_num_of_words_temp) / len(list_of_num_of_words_temp) # menghitung rata rata tf
                log_avg_tf = math.log10(avg_tf) # menghitung nilai log dari rata rata tf
                
                # looping untuk menghitung nilai tf yang baru
                list_of_num_of_words_temp_2 = [] # list tf untuk menyimpan hasil yang baru
                for tf in list_of_num_of_words_temp: # looping setiap tf
                    if tf == 0: # jika tf adalah 0 maka tidak perlu menghitung nilai log dari 0
                        list_of_num_of_words_temp_2.append(1  / (1 + log_avg_tf ))
                    else: # jika nilai tf bukan 0
                        list_of_num_of_words_temp_2.append( (1 + math.log10(tf))  / (1 + log_avg_tf ) )  
                #list_of_num_of_words_temp = [ ( (1 + math.log10(x))  / (1 + log_avg_tf ) )  for x in list_of_num_of_words_temp] # menghitung tf dengan log ave
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df_document = pd.concat([self.df_document, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df_document['tf_' + file_name] = list_of_num_of_words_temp_2 # menambahkan tf pada dokumen sekarang ke dataframe
        
        # menambahkan kolom dft pada data frame
        list_of_dft_temp = []
        for index, row in self.df_document.iterrows(): # looping setiap daftar kata pada dataframe
            curr_word = row['word'] # mendapatkan kata pada baris sekarang
            list_of_dft_temp.append(dft_temp[row['word']]) # menambahkan kemunculan kata sekarang ada di berapa dokumen ke dalam list
        self.df_document['dft_ori'] = list_of_dft_temp # menambahkan kolom dft
        
        
    def generate_tf_idf_document(self, df_type, normalization_type):
        
        # generate dft
        temp_dft = [] # tempat sementara untuk menyimpan dft
        if df_type == 'no': # menggunakan no idf
            temp_dft = self.df_document['dft_ori'].to_list() # mengisi temp_dft dengan dft yang sudah ada sebelumnya
        elif df_type == 'idf': # menggunakan idf
            temp_dft = self.df_document['dft_ori'].to_list() # mengambil dft sebelumnya
            
            # looping untuk menghitung nilai dft yang baru
            temp_dft_2 = [] # list sementara untuk menyimpan hasil perhitungan nilai dft yang baru
            for dft in temp_dft: # looping setiap dft
                if dft == 0: # jika nilai dft adalah 0 maka langsung menambahkan nilai log nya adalah 0
                    temp_dft_2.append(0)
                else: # jika nilai dft bukan 0 maka menghitung nilai log nya
                    temp_dft_2.append(math.log10(self.document_num / dft))
            #temp_dft = [math.log10(self.document_num / dft)  for dft in temp_dft] # mengganti dft dengan idf dft
            temp_dft = temp_dft_2 # mengganti list nilai dft yang lama dengan yang baru
        else: # menggunakan prob idf
            temp_dft = self.df_document['dft_ori'].to_list() # mengambil dft sebelumnya
            
            # looping untuk menghitung nilai dft yang baru
            temp_dft_2 = [] # list sementara untuk menyimpan hasil perhitungan nilai dft yang baru
            for dft in temp_dft: # looping setiap dft
                if dft == 0: # jika nilai dft adalah 0 maka langsung menambahkan nilai log nya adalah 0
                    temp_dft_2.append(0)
                else: # jika nilai dft bukan 0 maka menghitung nilai prob idf nya
                    temp_dft_2.append(max(0, math.log10(self.document_num - dft/ dft)) )
            
            #temp_dft = [max(0, math.log10(self.document_num - dft/ dft))  for dft in temp_dft] # mengganti dft dengan prob idf
            temp_dft = temp_dft_2 # mengganti list nilai dft yang lama dengan yang baru
        self.df_document['dft_doc'] = temp_dft # mengganti dft dengan tipe dft yang digunakan sebelumnya
        
        # generate tf idf original
        curr_file = os.listdir(self.path) # mendapatkan list file pada directory
        for file_name in curr_file: # looping setiap dokumen
            temp_df = pd.DataFrame({'tf_idf_' + file_name : self.df_document['tf_' + file_name] * self.df_document['dft_doc']}) # membuat kolom tf idf untuk setiap dokumen pada dataframe sementara
            self.df_document = pd.concat([self.df_document, temp_df], axis=1) # menggabungkan kolom tf idf untuk setiap dokumen
            #self.df_document['tf_idf_' + file_name] = self.df_document['tf_' + file_name] * self.df_document['dft_doc'] # membuat kolom tf idf untuk setiap dokumen
            
            
        # generate tf idf normalization
        if normalization_type == 'none': # menggunakan tipe normalization none
            pass
        else: # menggunakan tipe normalization cosine
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                curr_tf_idf = self.df_document['tf_idf_' + file_name].to_list()
                euclidean_norm = np.linalg.norm(curr_tf_idf) # menghitung euclidean norm
                if euclidean_norm != 0: # jika panjang euclidean bukan 0, maka dapat menggunakan persamaan
                    curr_tf_idf = [(tf_idf / euclidean_norm) for tf_idf in curr_tf_idf] # menghitung cosine normalization
                else: # jika panjang euclidean adalah 0, maka nilai tf_idf adalah 0
                    curr_tf_idf = [0 for tf_idf in curr_tf_idf]
                #temp_df = pd.DataFrame({'tf_idf_' + file_name : curr_tf_idf}) # membuat nilai tf idf dengan cosine normalization pada dataframe baru
                #self.df_document = pd.concat([self.df_document, temp_df], axis=1) # menggabungkan nilai tf idf dengan cosine normalization
                self.df_document['tf_idf_' + file_name] = curr_tf_idf # mengganti nilai tf idf dengan cosine normalization
    
    """
    def get_all_words_query(self): # mendapatkan kata yang tidak ada di document tapi ada di query
        query_curr = self.query # mengambil query
        query_curr = query_curr.lower() # mengubah query menjadi huruf kecil
        query_curr = ''.join(re.findall(r'[a-zA-Z]', query_curr)) # hanya mengabil huruf latin dari query
        tokens = query_curr.split(" ") # melakukan tokenisasi dari query
        words_curr = self.df_document['word'].to_list()
        
        for token in tokens: # looping setiap token dari query
            if (token not in words_curr) and (token not in self.words_not_in_doc): # jika token tidak ada di daftar kata dari dokumen dan di list words_not_in_doc
                self.words_not_in_doc.append(token) # menyimpan kata yang ada di query yang tidak ada di dokumen
        
        # menambahkan kata baru yang tidak ada sebelumnya ke dataframe
        self.list_of_words = self.df_document['words'].to_list() # mengambil daftar kata sekarang yang ada di data frame
        self.list_of_words.append(self.words_not_in_doc) # menambahkan kata yang tidak ada ke daftar kata
        self.df_document['words'] = self.list_of_words # mengganti kolom word pada data frame dengan daftar kata yang baru
        
        # mengisi kolom NaN dengan 0 karena pada dokumen tidak ada kata yang baru
        self.df_document = self.df_document.fillna(0)
    """
    
    def generate_tf_query(self, tf_type):
        query_curr = self.query # mengambil query
        query_curr = query_curr.lower() # mengubah query menjadi huruf kecil
        query_curr = re.sub(r'[^a-zA-Z\s]', '', query_curr) # hanya mengabil huruf latin dari query
        tokens = query_curr.split(" ") # melakukan tokenisasi dari query
        
        if tf_type == 'natural': # jika menggunakan tf natural
            list_of_num_of_words_temp = []
            for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan jumlah kata sekarang pada query ke dalam list
            self.df_document['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
        elif tf_type == 'logarithm':
            list_of_num_of_words_temp = []
            for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                
                if num_of_word == 0: # jika num_of_word adalah 0 maka nilai tf tf adalah 1
                    num_of_word = 1
                else: # num_of_word bukan 0
                    num_of_word = 1 + math.log10(num_of_word) # menghitung tf menggunakan logarithm
                
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
            self.df_document['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
        
        elif tf_type == 'augmented':
            list_of_num_of_words_temp = []
            for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada query ke dalam list
            
            maximum_num_of_words = max(list_of_num_of_words_temp) # menghitung nilai maximum tf
            list_of_num_of_words_temp = [(0.5 + ( (0.5 * x)  / maximum_num_of_words ) ) for x in list_of_num_of_words_temp] # menghitung tf dengan augmented
            self.df_document['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
            
        elif tf_type == 'boolean':
            list_of_num_of_words_temp = []
            for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada query ke dalam list
            
            list_of_num_of_words_temp = [1 if x != 0 else 0 for x in list_of_num_of_words_temp] # menghitung tf dengan boolean
            self.df_document['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
            
        else: # log ave
            list_of_num_of_words_temp = []
            for index, row in self.df_document.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada query ke dalam list
            
            avg_tf = sum(list_of_num_of_words_temp) / len(list_of_num_of_words_temp) # menghitung rata rata tf
            log_avg_tf = math.log10(avg_tf) # menghitung nilai log dari rata rata tf
            
            # looping untuk menghitung nilai tf yang baru
            list_of_num_of_words_temp_2 = [] # list tf untuk menyimpan hasil yang baru
            for tf in list_of_num_of_words_temp: # looping setiap tf
                if tf == 0: # jika tf adalah 0 maka tidak perlu menghitung nilai log dari 0
                    list_of_num_of_words_temp_2.append(1  / (1 + log_avg_tf ))
                else: # jika nilai tf bukan 0
                    list_of_num_of_words_temp_2.append( (1 + math.log10(tf))  / (1 + log_avg_tf ) )  
            
            self.df_document['tf_query'] = list_of_num_of_words_temp_2 # menambahkan tf pada query ke dataframe
        
    def generate_tf_idf_query(self, df_type, normalization_type):
        # generate dft
        temp_dft = [] # tempat sementara untuk menyimpan dft
        if df_type == 'no': # menggunakan no idf
            temp_dft = self.df_document['dft_ori'].to_list() # mengisi temp_dft dengan dft yang sudah ada sebelumnya
        elif df_type == 'idf': # menggunakan idf
            temp_dft = self.df_document['dft_ori'].to_list() # mengambil dft sebelumnya
            
            # looping untuk menghitung nilai dft yang baru
            temp_dft_2 = [] # list sementara untuk menyimpan hasil perhitungan nilai dft yang baru
            for dft in temp_dft: # looping setiap dft
                if dft == 0: # jika nilai dft adalah 0 maka langsung menambahkan nilai log nya adalah 0
                    temp_dft_2.append(0)
                else: # jika nilai dft bukan 0 maka menghitung nilai log nya
                    temp_dft_2.append(math.log10(self.document_num / dft))
            #temp_dft = [math.log10(self.document_num / dft)  for dft in temp_dft] # mengganti dft dengan idf dft
            temp_dft = temp_dft_2 # mengganti list nilai dft yang lama dengan yang baru
        else: # menggunakan prob idf
            temp_dft = self.df_document['dft_ori'].to_list() # mengambil dft sebelumnya
            
            # looping untuk menghitung nilai dft yang baru
            temp_dft_2 = [] # list sementara untuk menyimpan hasil perhitungan nilai dft yang baru
            for dft in temp_dft: # looping setiap dft
                if dft == 0: # jika nilai dft adalah 0 maka langsung menambahkan nilai log nya adalah 0
                    temp_dft_2.append(0)
                else: # jika nilai dft bukan 0 maka menghitung nilai prob idf nya
                    temp_dft_2.append(max(0, math.log10(self.document_num - dft/ dft)) )
            
            #temp_dft = [max(0, math.log10(self.document_num - dft/ dft))  for dft in temp_dft] # mengganti dft dengan prob idf
            temp_dft = temp_dft_2 # mengganti list nilai dft yang lama dengan yang baru
        self.df_document['dft_query'] = temp_dft # mengganti dft dengan tipe dft yang digunakan sebelumnya
        
        # generate tf idf original
        self.df_document['tf_idf_query'] = self.df_document['tf_query'] * self.df_document['dft_query'] # membuat kolom tf idf untuk query
    
        # generate tf idf normalization
        if normalization_type == 'none': # menggunakan tipe normalization none
            pass
        else: # menggunakan tipe normalization cosine
            curr_tf_idf = self.df_document['tf_idf_query'].to_list()
            euclidean_norm = np.linalg.norm(curr_tf_idf) # menghitung euclidean norm
            if euclidean_norm != 0: # jika panjang euclidean bukan 0, maka dapat menggunakan persamaan
                curr_tf_idf = [(tf_idf / euclidean_norm) for tf_idf in curr_tf_idf] # menghitung cosine normalization
            else: # jika panjang euclidean bukan 0, maka nilai tf idf adalah 0
                curr_tf_idf = [0 for tf_idf in curr_tf_idf] # nilai tf idf adalah 0
            self.df_document['tf_idf_query'] = curr_tf_idf # mengganti nilai tf idf dengan cosine normalization

    def similarity(self):
        self.generate_document_index()
        self.generate_query_index()
        sim_dict = {}
        curr_file = os.listdir(self.path) # mendapatkan list file pada directory
        for file_name in curr_file: # looping setiap dokumen
            tf_idf_doc_temp = self.df_document['tf_idf_' + file_name].to_list() # mendapatkan tf idf untuk dokumen sekarang
            tf_idf_query_temp = self.df_document['tf_idf_query'].to_list() # mendapatkan tf idf untuk query
            sim = cosine_similarity([tf_idf_doc_temp], [tf_idf_query_temp]) # menghitung kesamaan antara dokumen sekarang dengan query
            sim_dict[file_name] = sim[0][0] # menyimpan nilai kesamaan antara dokumen sekarang dengan query
         
        sorted_doc = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True) # mengurutkan kesamaan dokumen dengan query by decreasing order
        print("{:^20} : {:^10}".format('File Name', 'Similarity'))
        for key, value in sorted_doc: # looping untuk print kesamaan dokumen dengan query
             #print(key, ":", value) # print kesamaan dokumen dengan query
             print("{:<20} : {:<10}".format(key, value))
        return sorted_doc[0]
    
def main():
    path = input("Enter your document path:")
    tf_type = ['natural', 'logarithm', 'augmented', 'boolean', 'log_ave']
    dft_type = ['no', 'idf', 'prob_idf']
    normalization_type = ['none', 'cosine']
    
    #vsm = VSM(path, 'natural', 'no', 'none')
    num = 0
    
    query = input("Enter your query:")
    
    for tf in tf_type:
        for dft in dft_type:
            for normalization in normalization_type:
                for tf_2 in tf_type:
                    for dft_2 in dft_type:
                        for normalization_2 in normalization_type:
                            print(tf, dft, normalization, tf_2, dft_2, normalization_2)
                            vsm = VSM(path, tf, dft, normalization)
                            vsm.set_query(query,tf_2, dft_2, normalization_2 )
                            print("Your query is under process...")
                            res = vsm.similarity()
                            if res == 'document_405.txt':
                                num += 1
    
    print(num)
    #vsm.set_query(query,'natural', 'no', 'none' )
    #print("Your query is under process...")
    #vsm.similarity()

if __name__=="__main__": 
    main()
    
    
    
    
    
    