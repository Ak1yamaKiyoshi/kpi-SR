import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QListWidget
from PyQt5.QtCore import Qt
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        self.loadFiles()
        self.cacheEmbeddings()

    def initUI(self):
        layout = QVBoxLayout()

        self.textInput = QTextEdit()
        layout.addWidget(self.textInput)

        self.searchButton = QPushButton('Search')
        self.searchButton.clicked.connect(self.performSearch)
        layout.addWidget(self.searchButton)

        self.resultsList = QListWidget()
        layout.addWidget(self.resultsList)

        self.setLayout(layout)
        self.setWindowTitle('Text Search')
        self.setGeometry(300, 300, 600, 500)

    def loadModel(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def loadFiles(self):
        self.text_pieces = []
        for filename in os.listdir('./defend'):
            if filename.endswith('.txt'):
                with open(os.path.join('./defend', filename), 'r') as file:
                    lines = file.read()
                    lines.split("\n\n")
                    
                    #lines = file.readlines()
                    self.text_pieces.extend([line.strip() for line in lines if line.strip()])

    def getEmbedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def cacheEmbeddings(self):
        self.cached_embeddings = []
        for piece in self.text_pieces:
            self.cached_embeddings.append(self.getEmbedding(piece))
        self.cached_embeddings = np.array(self.cached_embeddings)

    def performSearch(self):
        query = self.textInput.toPlainText()
        query_embedding = self.getEmbedding(query)

        similarities = cosine_similarity([query_embedding], self.cached_embeddings)[0]
        
        top_indices = similarities.argsort()[-5:][::-1]
        top_matches = [(self.text_pieces[i], similarities[i]) for i in top_indices]

        self.resultsList.clear()
        for piece, similarity in top_matches:
            self.resultsList.addItem(f"{piece} (Similarity: {similarity:.4f})")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TextSearchApp()
    ex.show()
    sys.exit(app.exec_())