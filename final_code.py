#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:25:59 2024

@author: williamnehemia
"""

# import library
import os
import re
import math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




class VSM:
    def __init__(self, path, tf_doc_type, dft_type, normalization_type):
        self.path = path # menyimpan path dokumen
        self.document_num = len(os.listdir(path)) # menyimpan banyak dokumen pada directory
        self.list_of_words = [] # menyimpan daftar semua kata dari dokumen + query
        self.df = pd.DataFrame() # dataframe untuk menyimpan tf, dft, idf, tf idf dari dokumen dan query
        self.tf_doc_type = tf_doc_type # menyimpan tipe tf dokumen (natural / logarithm / augmented / boolean / log_ave)
        self.dft_type = dft_type # menyimpan tipe dft dokumen (no / idf / prob_idf)
        self.normalization_type = normalization_type # menyimpan tipe normalization dokumen (none / cosine)

    
    def set_query(self, query, tf_query_type, dft_query_type, normalization_query_type):
        self.query = query # menyimpan query
        self.tf_query_type = tf_query_type # menyimpan tipe tf query (natural / logarithm / augmented / boolean / log_ave)
        self.dft_query_type = dft_query_type # menyimpan tipe dft query (no / idf / prob_idf)
        self.normalization_query_type = normalization_query_type # menyimpan tipe normalization query (none / cosine)
    
    def generate_document_index(self):
        # mendapatkan  semua kata / term dari dokumen dan query
        self.get_all_words()
        
        # generate tf document
        self.generate_tf_document(self.tf_doc_type)
        
        # generate tf - idf document
        self.generate_tf_idf_document(self.dft_type, self.normalization_type)
    
    def generate_query_index(self):
        # generate tf query
        self.generate_tf_query(self.tf_query_type)
        # generate tf idf query
        self.generate_tf_idf_query(self.dft_query_type, self.normalization_query_type)
    
    def get_all_words(self): # method untuk mendapatkan  semua kata / term dari dokumen dan query
        curr_file = os.listdir(self.path) # mendapatkan list file pada directory
        for file_name in curr_file: # looping setiap dokumen
            with open(self.path + '/' + file_name, "r") as file:
                content = file.read() # baca konten pada dokumen
                
                content = content.lower() # mengubah huruf menjadi huruf kecil
                content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                tokens = content.split(" ") # split dokumen dengan spasi
                
                # looping per token dalam dokumen
                for token in tokens:
                    if token not in self.list_of_words: # jika token tidak ada dalam daftar kata
                        self.list_of_words.append(token) # menambahkan token ke list daftar kata
                        
        query_curr = self.query # mengambil query
        query_curr = query_curr.lower() # mengubah query menjadi huruf kecil
        query_curr = re.sub(r'[^a-zA-Z\s]', '', query_curr) # hanya mengabil huruf latin dari query
        tokens = query_curr.split(" ") # melakukan tokenisasi dari query
        for token in tokens: # looping per token
            if token not in self.list_of_words: # jika token dari query tidak ada dalam daftar kata
                self.list_of_words.append(token) # menambahkan token ke daftar kata
            
        
        self.df['word'] = self.list_of_words # membuat kolom word pada dataframe
    
    def generate_tf_document(self, tf_type): # method untuk generate tf document
        dft_temp = {key: 0 for key in self.list_of_words} # membuat dictionary untuk menyimpan jumlah kemunculan dokumen ada di berapa dokumen (dft)
        if tf_type == 'natural': # jika menggunakan tf natural
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada dokumen sekarang
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ") # split konten dengan spasi
                    for index, row in self.df.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan jumlah kata sekarang pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df = pd.concat([self.df, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
                        
                
        elif tf_type == 'logarithm': # jika menggunakan tf logarithm
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada dokumen sekarang
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ") # split konten dengan spasi
                    for index, row in self.df.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        
                        if num_of_word == 0: # jika num_of_word adalah 0 maka tidak akan dihitung nilai log nya
                            num_of_word = 1
                        else: # num_of_word bukan 0
                            num_of_word = 1 + math.log10(num_of_word) # menghitung tf menggunakan logarithm
                        
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if tokens.count(curr_word) > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df = pd.concat([self.df, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
        elif tf_type == 'augmented': # jika menggunakan tf augmented
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada dokumen sekarang
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ") # split konten dengan spasi
                    for index, row in self.df.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                maximum_num_of_words = max(list_of_num_of_words_temp) # mencari nilai maximum tf
                list_of_num_of_words_temp = [(0.5 + ( (0.5 * x)  / maximum_num_of_words ) ) for x in list_of_num_of_words_temp] # menghitung tf dengan augmented
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df = pd.concat([self.df, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
        elif tf_type == 'boolean': # jika menggunakan tf boolean
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada dokumen sekarang
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ") # split konten dengan spasi
                    for index, row in self.df.iterrows(): # looping setiap daftar kata
                        curr_word = row['word'] # mendapatkan kata pada baris sekarang
                        num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada dokumen
                        list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
                        if num_of_word > 0: # jika jumlah kemunculan kata lebih dari 0
                            dft_temp[curr_word] += 1 # menambahkan 1 ke kemunculan kata tersebut karena muncul di dokumen ini
                list_of_num_of_words_temp = [1 if x != 0 else 0 for x in list_of_num_of_words_temp] # menghitung tf dengan boolean
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df = pd.concat([self.df, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df['tf_' + file_name] = list_of_num_of_words_temp # menambahkan tf pada dokumen sekarang ke dataframe
        else:  # jika menggunakan tf log ave
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada dokumen sekarang
                with open(self.path + '/' + file_name, "r") as file:
                    content = file.read() # baca konten pada dokumen
                    content = content.lower() # mengubah huruf menjadi huruf kecil
                    content = re.sub(r'[^a-zA-Z\s]', '', content) # hanyak mengambil huruf 
                    tokens = content.split(" ") # split konten dengan spasi
                    for index, row in self.df.iterrows(): # looping setiap daftar kata
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
                df_temp = pd.DataFrame({'tf_' + file_name : list_of_num_of_words_temp_2 }) # membuat dataframe sementara untuk menyimpan tf pada dokumen sekarang
                self.df = pd.concat([self.df, df_temp], axis=1) # menambahkan tf pada dokumen sekarang ke dataframe
                #self.df['tf_' + file_name] = list_of_num_of_words_temp_2 # menambahkan tf pada dokumen sekarang ke dataframe
        
        # menambahkan kolom dft pada data frame
        list_of_dft_temp = []
        for index, row in self.df.iterrows(): # looping setiap daftar kata pada dataframe
            curr_word = row['word'] # mendapatkan kata pada baris sekarang
            list_of_dft_temp.append(dft_temp[row['word']]) # menambahkan kemunculan kata sekarang ada di berapa dokumen ke dalam list
        self.df['dft_ori'] = list_of_dft_temp # menambahkan kolom dft
        
        
    def generate_tf_idf_document(self, df_type, normalization_type): # method untuk generate tf idf dari dokumen
        
        # generate dft
        temp_dft = [] # tempat sementara untuk menyimpan dft
        if df_type == 'no': # menggunakan no idf
            temp_dft = self.df['dft_ori'].to_list() # mengisi temp_dft dengan dft yang sudah ada sebelumnya
        elif df_type == 'idf': # menggunakan idf
            temp_dft = self.df['dft_ori'].to_list() # mengambil dft sebelumnya
            
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
            temp_dft = self.df['dft_ori'].to_list() # mengambil dft sebelumnya
            
            # looping untuk menghitung nilai dft yang baru
            temp_dft_2 = [] # list sementara untuk menyimpan hasil perhitungan nilai dft yang baru
            for dft in temp_dft: # looping setiap dft
                if dft == 0: # jika nilai dft adalah 0 maka langsung menambahkan nilai log nya adalah 0
                    temp_dft_2.append(0)
                else: # jika nilai dft bukan 0 maka menghitung nilai prob idf nya
                    temp_dft_2.append(max(0, math.log10(self.document_num - dft/ dft)) )
            
            #temp_dft = [max(0, math.log10(self.document_num - dft/ dft))  for dft in temp_dft] # mengganti dft dengan prob idf
            temp_dft = temp_dft_2 # mengganti list nilai dft yang lama dengan yang baru
        self.df['dft_doc'] = temp_dft # mengganti dft dengan tipe dft yang digunakan sebelumnya
        
        # generate tf idf original
        curr_file = os.listdir(self.path) # mendapatkan list file pada directory
        for file_name in curr_file: # looping setiap dokumen
            temp_df = pd.DataFrame({'tf_idf_' + file_name : self.df['tf_' + file_name] * self.df['dft_doc']}) # membuat kolom tf idf untuk setiap dokumen pada dataframe sementara
            self.df = pd.concat([self.df, temp_df], axis=1) # menggabungkan kolom tf idf untuk setiap dokumen
            #self.df['tf_idf_' + file_name] = self.df['tf_' + file_name] * self.df['dft_doc'] # membuat kolom tf idf untuk setiap dokumen
            
            
        # generate tf idf normalization
        if normalization_type == 'none': # menggunakan tipe normalization none
            pass
        else: # menggunakan tipe normalization cosine
            curr_file = os.listdir(self.path) # mendapatkan list file pada directory
            for file_name in curr_file: # looping setiap dokumen
                curr_tf_idf = self.df['tf_idf_' + file_name].to_list()
                euclidean_norm = np.linalg.norm(curr_tf_idf) # menghitung euclidean norm
                if euclidean_norm != 0: # jika panjang euclidean bukan 0, maka dapat menggunakan persamaan
                    curr_tf_idf = [(tf_idf / euclidean_norm) for tf_idf in curr_tf_idf] # menghitung cosine normalization
                else: # jika panjang euclidean adalah 0, maka nilai tf_idf adalah 0
                    curr_tf_idf = [0 for tf_idf in curr_tf_idf]
                #temp_df = pd.DataFrame({'tf_idf_' + file_name : curr_tf_idf}) # membuat nilai tf idf dengan cosine normalization pada dataframe baru
                #self.df = pd.concat([self.df, temp_df], axis=1) # menggabungkan nilai tf idf dengan cosine normalization
                self.df['tf_idf_' + file_name] = curr_tf_idf # mengganti nilai tf idf dengan cosine normalization
    
    
    
    def generate_tf_query(self, tf_type): # method untuk generate tf pada query
        query_curr = self.query # mengambil query
        query_curr = query_curr.lower() # mengubah query menjadi huruf kecil
        query_curr = re.sub(r'[^a-zA-Z\s]', '', query_curr) # hanya mengabil huruf latin dari query
        tokens = query_curr.split(" ") # melakukan tokenisasi dari query
        
        if tf_type == 'natural': # jika menggunakan tf natural
            list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada query
            for index, row in self.df.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan jumlah kata sekarang pada query ke dalam list
            self.df['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
        elif tf_type == 'logarithm':
            list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada query
            for index, row in self.df.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                
                if num_of_word == 0: # jika num_of_word adalah 0 maka nilai tf tf adalah 1
                    num_of_word = 1
                else: # num_of_word bukan 0
                    num_of_word = 1 + math.log10(num_of_word) # menghitung tf menggunakan logarithm
                
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada dokumen ke dalam list
            self.df['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
        
        elif tf_type == 'augmented':
            list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada query
            for index, row in self.df.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada query ke dalam list
            
            maximum_num_of_words = max(list_of_num_of_words_temp) # menghitung nilai maximum tf
            list_of_num_of_words_temp = [(0.5 + ( (0.5 * x)  / maximum_num_of_words ) ) for x in list_of_num_of_words_temp] # menghitung tf dengan augmented
            self.df['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
            
        elif tf_type == 'boolean':
            list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada query
            for index, row in self.df.iterrows(): # looping setiap daftar kata
                curr_word = row['word'] # mendapatkan kata pada baris sekarang
                num_of_word = tokens.count(curr_word) # menghitung jumlah kata sekarang pada query
                list_of_num_of_words_temp.append(num_of_word) # menambahkan tf pada query ke dalam list
            
            list_of_num_of_words_temp = [1 if x != 0 else 0 for x in list_of_num_of_words_temp] # menghitung tf dengan boolean
            self.df['tf_query'] = list_of_num_of_words_temp # menambahkan tf pada query ke dataframe
            
        else: # log ave
            list_of_num_of_words_temp = [] # list sementara untuk menyimpan kemunculan kata pada query
            for index, row in self.df.iterrows(): # looping setiap daftar kata
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
            
            self.df['tf_query'] = list_of_num_of_words_temp_2 # menambahkan tf pada query ke dataframe
        
    def generate_tf_idf_query(self, df_type, normalization_type): # method untuk generate tf idf pada query
        # generate dft
        temp_dft = [] # tempat sementara untuk menyimpan dft
        if df_type == 'no': # menggunakan no idf
            temp_dft = self.df['dft_ori'].to_list() # mengisi temp_dft dengan dft yang sudah ada sebelumnya
        elif df_type == 'idf': # menggunakan idf
            temp_dft = self.df['dft_ori'].to_list() # mengambil dft sebelumnya
            
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
            temp_dft = self.df['dft_ori'].to_list() # mengambil dft sebelumnya
            
            # looping untuk menghitung nilai dft yang baru
            temp_dft_2 = [] # list sementara untuk menyimpan hasil perhitungan nilai dft yang baru
            for dft in temp_dft: # looping setiap dft
                if dft == 0: # jika nilai dft adalah 0 maka langsung menambahkan nilai log nya adalah 0
                    temp_dft_2.append(0)
                else: # jika nilai dft bukan 0 maka menghitung nilai prob idf nya
                    temp_dft_2.append(max(0, math.log10(self.document_num - dft/ dft)) )
            
            #temp_dft = [max(0, math.log10(self.document_num - dft/ dft))  for dft in temp_dft] # mengganti dft dengan prob idf
            temp_dft = temp_dft_2 # mengganti list nilai dft yang lama dengan yang baru
        self.df['dft_query'] = temp_dft # mengganti dft dengan tipe dft yang digunakan sebelumnya
        
        # generate tf idf original
        self.df['tf_idf_query'] = self.df['tf_query'] * self.df['dft_query'] # membuat kolom tf idf untuk query
    
        # generate tf idf normalization
        if normalization_type == 'none': # menggunakan tipe normalization none
            pass
        else: # menggunakan tipe normalization cosine
            curr_tf_idf = self.df['tf_idf_query'].to_list()
            euclidean_norm = np.linalg.norm(curr_tf_idf) # menghitung euclidean norm
            if euclidean_norm != 0: # jika panjang euclidean bukan 0, maka dapat menggunakan persamaan
                curr_tf_idf = [(tf_idf / euclidean_norm) for tf_idf in curr_tf_idf] # menghitung cosine normalization
            else: # jika panjang euclidean bukan 0, maka nilai tf idf adalah 0
                curr_tf_idf = [0 for tf_idf in curr_tf_idf] # nilai tf idf adalah 0
            self.df['tf_idf_query'] = curr_tf_idf # mengganti nilai tf idf dengan cosine normalization

    def similarity(self): # method untuk menghitung kesamaan antara query dengan setiap dokumen dan mengurutkannya berdasarkan kemiripan paling tinggi
        self.generate_document_index() # generate document index
        self.generate_query_index() # generate query index
        sim_dict = {} # dictionary sementara untuk menyimpan kesamaan antara query dengan setiap dokumen
        #pd.set_option('display.max_columns', None)
        #print(self.df.head())
        #print(self.df['dft_query'].to_string(index=False))
        curr_file = os.listdir(self.path) # mendapatkan list file pada directory
        for file_name in curr_file: # looping setiap dokumen
            tf_idf_doc_temp = self.df['tf_idf_' + file_name].to_list() # mendapatkan tf idf untuk dokumen sekarang
            tf_idf_query_temp = self.df['tf_idf_query'].to_list() # mendapatkan tf idf untuk query
            sim = cosine_similarity([tf_idf_doc_temp], [tf_idf_query_temp]) # menghitung kesamaan antara dokumen sekarang dengan query
            sim_dict[file_name] = sim[0][0] # menyimpan nilai kesamaan antara dokumen sekarang dengan query

        sorted_doc = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True) # mengurutkan kesamaan dokumen dengan query by decreasing order
        
        print("{:^20} : {:^10}".format('File Name', 'Similarity'))
        for key, value in sorted_doc: # looping untuk print kesamaan dokumen dengan query
             #print(key, ":", value) # print kesamaan dokumen dengan query
             print("{:<20} : {:<10}".format(key, value))
        return sorted_doc[0]

def start_tf_doc():
    print("Choose the type of term frequency of document by typing the number (ex: 1)")
    print("1. Natural")
    print("2. Logarithm")
    print("3. Augmented")
    print("4. Boolean")
    print("5. Log Ave")
    user_input =  int(input())
    if user_input == 1:
        return "natural"
    elif user_input == 2:
        return "logarithm"
    elif user_input == 3:
        return "augmented"
    elif user_input == 4:
        return "boolean"
    else:
        return "log_ave"
    
def start_tf_query():
    print("Choose the type of term frequency of query by typing the number (ex: 1)")
    print("1. Natural")
    print("2. Logarithm")
    print("3. Augmented")
    print("4. Boolean")
    print("5. Log Ave")
    user_input =  int(input())
    if user_input == 1:
        return "natural"
    elif user_input == 2:
        return "logarithm"
    elif user_input == 3:
        return "augmented"
    elif user_input == 4:
        return "boolean"
    else:
        return "log_ave"
    
def start_df_doc():
    print("Choose the type of document frequency of document by typing the number (ex: 1)")
    print("1. No Idf")
    print("2. Idf")
    print("3. Prob Idf")
    user_input =  int(input())
    if user_input == 1:
        return "no"
    elif user_input == 2:
        return "idf"
    else:
        return "prob_idf"
    
def start_df_query():
    print("Choose the type of document frequency of query by typing the number (ex: 1)")
    print("1. No Idf")
    print("2. Idf")
    print("3. Prob Idf")
    user_input =  int(input())
    if user_input == 1:
        return "no"
    elif user_input == 2:
        return "idf"
    else:
        return "prob_idf"

def start_normalization_doc():
    print("Choose the type of normalization of document by typing the number (ex: 1)")
    print("1. None")
    print("2. Cosine")
    user_input =  int(input())
    if user_input == 1:
        return "none"
    else:
        return "cosine"
    
def start_normalization_query():
    print("Choose the type of normalization of query by typing the number (ex: 1)")
    print("1. None")
    print("2. Cosine")
    user_input =  int(input())
    if user_input == 1:
        return "none"
    else:
        return "cosine"

def main():
    
    # input untuk tipe tf, dft, dan normalisasi untuk dokumen dan query
    tf_doc = start_tf_doc()
    tf_query = start_tf_query()
    df_doc = start_df_doc()
    df_query = start_df_query()
    norm_doc = start_normalization_doc()
    norm_query = start_normalization_query()

    
    # input path directory
    path = input("Enter your document path:")
    
    
    # membuat object class VSM
    vsm = VSM(path, tf_doc, df_doc, norm_doc)
    
   
    # input query
    query = input("Enter your query:")
    
    # memasukkan query pada object VSM
    vsm.set_query(query,tf_query, df_query, norm_query )
   
    
    
    
    
    print("Your query is under process...")
    
    # pemeringkatan dokumen dan nilai relevansinya
    vsm.similarity()
    
    
    
   
    
    

if __name__=="__main__": 
    main()
    
    
    
    
    
    