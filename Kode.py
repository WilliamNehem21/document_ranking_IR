#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:25:59 2024

@author: williamnehemia
"""

import os
import re
import math

if __name__=="__main__": 
    path = input("Masukkan path dokumen:")
    query = input("Masukkan query pencarian:")

class VSM:
    def __init__(self, path, query):
        self.path = path
        self.query = query
        self.document_num = len(os.listdir(path))
        self.words_document = {}
        self.words_query = {}
        self.dft = {}
        self.idf = {}
        self.tf_idf_doc = {}
        self.tf_idf_query = {}
        self.list_of_words = []
        self.ranked_document = {}
        self.tf_doc = {}
    
        
    
    def generate_vector_document(self):
        curr_file = os.listdir(self.path)
        for file_name in curr_file:
            words = {} # dictionary sementara untuk menyimpan jumlah kemunculan kata per dokumen
            with open(self.path + file_name, "r") as file:
                content = file.read()
                content = content.lower()
                content = ''.join(re.findall(r'[a-zA-Z]', content))
                tokens = content.split(" ")
                
                # looping per token dalam dokumen
                for token in tokens:
                    
                    # menambahkan jumlah kemunculan kata ke dalam dictionary sementara
                    if token not in words:
                        words[token] = 1
                    else:
                        words[token] += 1
                    
                    # untuk menyimpan semua kata
                    if token not in self.list_of_words:
                        self.list_of_words.append(token)
                
                # looping semua kata dalam dokumen untuk menghitung kemunculan kata dalam dokumen
                for word in words:
                    if word not in self.dft:
                        self.dft[word] = 1
                    else:
                        self.dft[word] += 1
            self.words_document[file_name] = words
            
        # memindahkan kemunculan kata per dokumen ke dictionary yang menyimpan semua kemunculan kata dari setiap dokumen
        for file in self.words_document: # looping setiap dokumen
            temp_words_doc = self.words_document[file] # mendapatkan jumlah kemunculan kata per dokumen
            temp_tf_doc = {} # dictionary sementara untuk menyimpan kemunculan semua kata per dokumen
            for word in self.list_of_words: # looping setiap dokumen 
                if word in temp_words_doc: # jika kata terdapat pada dokumen
                    temp_tf_doc[word] = temp_words_doc[word] # mengisi jumlah kemunculan kata dengan jumlah kemunculannya
                else: # jika kata tidak ada pada dokumen
                    temp_tf_doc[word] = 0 # mengisi jumlah kemunculan kata dengan 0
            self.tf_doc[file] = temp_tf_doc
    
    def generate_tf_idf_doc(self):
        # membuat idf
        for word in self.dft:
            value = self.dft[word]
            self.idf[word] = math.log10(self.document_num / value)
        
        # menghitung tf-idf document
        for doc in self.tf_doc: # looping setiap dokumen
            temp_tf = self.tf_doc[doc] # menyimpan sementara kemunculan kata per dokumen
            temp_tf_idf = {} # dictionary sementara untuk menyimpan tf idf per dokumen
            for word in temp_tf: # looping setiap kata dalam dokumen
                temp_tf_idf[word] = temp_tf[word] * self.idf[word] # menghitung tf idf per kata
            self.tf_idf_doc[doc] = temp_tf_idf # menyimpan hasil tf idf dari semua kata pada suatu dokumen ke dalam dictionary tf idf seluruh dokumen
            
                
    
    def generate_vector_query(self):
        query_curr = self.query
        query_curr = query_curr.lower()
        query_curr = ''.join(re.findall(r'[a-zA-Z]', query_curr))
        tokens = query_curr.split(" ")
        for token in tokens:
            if token not in self.words_query:
                self.words_query[token] = 1
            else:
                self.words_query[token] += 1
            
            # untuk menyimpan semua kata
            if token not in self.list_of_words:
                self.list_of_words.append(token)
    
    def generate_tf_idf_query(self):
        # menghitung tf-idf query
        for word in self.words_query:
            self.tf_idf_query[word] = self.words_query[word] * self.idf[word]
    
    def ranking_document(self):
        pass
    # bisa looping per kata pake all words, atau bikin jumlah tf dan tf idf nya itu list di dlm dokumennya, jadi harus di looping lg