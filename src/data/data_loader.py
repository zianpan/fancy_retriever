import json
import os
from typing import Dict, List, Any, Optional

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> List[Dict[str, Any]]:
        if self.data is None:
            try:
                with open(self.data_path, 'r') as f:
                    self.data = json.load(f)
                # Validate data structure
                if not isinstance(self.data, list):
                    print(f"Warning: Expected list, got {type(self.data)}")
                    if isinstance(self.data, dict):
                        # Convert to list if it's a dictionary
                        self.data = [self.data]
                
                # Ensure each entry has the required fields
                validated_data = []
                for item in self.data:
                    if not isinstance(item, dict):
                        print(f"Warning: Skipping non-dictionary item: {item}")
                        continue
                    
                    # Check for required fields
                    if 'question' not in item:
                        print(f"Warning: Item missing 'question' field: {item}")
                        continue
                    
                    if 'answers' not in item and 'answer' not in item:
                        print(f"Warning: Item missing answer field: {item}")
                        continue
                    
                    # Standardize answer field
                    if 'answer' in item and 'answers' not in item:
                        item['answers'] = item['answer']
                    
                    # Ensure contexts field exists
                    if 'ctxs' not in item:
                        print(f"Warning: Item missing 'ctxs' field: {item}")
                        item['ctxs'] = []
                    
                    validated_data.append(item)
                
                self.data = validated_data
                
                print(f"Successfully loaded {len(self.data)} items from {self.data_path}")
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                self.data = []
        
        return self.data
    
    def get_questions(self) -> List[str]:
        data = self.load_data()
        return [item['question'] for item in data]
    
    def get_answers(self) -> List[str]:
        data = self.load_data()
        return [item['answers'] for item in data]
    
    def get_contexts(self) -> List[List[Dict[str, Any]]]:
        data = self.load_data()
        return [item['ctxs'] for item in data]
    
    def get_item_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        data = self.load_data()
        if 0 <= index < len(data):
            return data[index]
        print(f"Index {index} out of range (0-{len(data)-1})")
        return None
    
    def get_item_by_question(self, question: str) -> Optional[Dict[str, Any]]:
        data = self.load_data()
        for item in data:
            if item['question'] == question:
                return item
        return None