import json
import re
import os
from typing import List, Dict, Any, Tuple

class DictionaryBasedTokenizer:
    
    def __init__(self, verbs_file, nouns_file, time_words_file, locations_file):
        # Tải các từ điển
        self.verbs = self._load_dictionary(verbs_file)
        self.nouns = self._load_dictionary(nouns_file)
        self.time_words = self._load_dictionary(time_words_file)
        self.locations = self._load_dictionary(locations_file)
        
        print(f"Đã tải {len(self.verbs)} động từ từ {verbs_file}")
        print(f"Đã tải {len(self.nouns)} danh từ từ {nouns_file}")
        print(f"Đã tải {len(self.time_words)} từ thời gian từ {time_words_file}")
        print(f"Đã tải {len(self.locations)} địa điểm từ {locations_file}")
    
    def _load_dictionary(self, file_path: str) -> List[str]:
        """Đọc danh sách từ từ file"""
        words = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # Bỏ qua dòng trống và comment
                        words.append(word)
            return words
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return []
    
    def __call__(self, text):
        """Xử lý văn bản và trả về đối tượng để phân tích POS"""
        tokens = text.split()
        return TokenizedDoc(tokens, self)


class TokenizedDoc:
    """Đối tượng chứa văn bản đã tokenize và hỗ trợ phân tích"""
    
    def __init__(self, tokens, tokenizer):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.ents = []  # Thực thể được nhận biết
        
        # Tạo các token với thông tin POS
        self._tokens = [AnnotatedToken(t, i, tokenizer) for i, t in enumerate(tokens)]
    
    def __iter__(self):
        return iter(self._tokens)
    
    def __len__(self):
        return len(self._tokens)
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._tokens[i]
        return self._tokens[i]


class AnnotatedToken:
    """Token với thông tin part-of-speech từ từ điển"""
    
    def __init__(self, text, i, tokenizer):
        self.text = text
        self.i = i
        self.tokenizer = tokenizer
        
        # Xác định POS dựa trên từ điển
        self.pos_ = self._determine_pos(text)
    
    def _determine_pos(self, text):
        """Xác định part-of-speech dựa trên từ điển"""
        text_lower = text.lower()
        
        # Kiểm tra text trong các từ điển
        for word in self.tokenizer.verbs:
            if text_lower == word.lower() or self._is_part_of_compound(text_lower, word.lower()):
                return "VERB"
        
        for word in self.tokenizer.nouns:
            if text_lower == word.lower() or self._is_part_of_compound(text_lower, word.lower()):
                return "NOUN"
        
        for word in self.tokenizer.time_words:
            if text_lower == word.lower() or self._is_part_of_compound(text_lower, word.lower()):
                return "TIME"
        
        # Kiểm tra nếu là số
        if text_lower.isdigit():
            return "NUM"
        
        return "X"  # Unknown
    
    def _is_part_of_compound(self, text, word):
        """Kiểm tra nếu token là một phần của từ ghép"""
        if " " in word:
            parts = word.split()
            return text in parts
        return False


class VietnameseEventTagger:
    def __init__(self, verbs_file, nouns_file, time_words_file, locations_file, triggers_file):
        # Tạo tokenizer với từ điển
        self.tokenizer = DictionaryBasedTokenizer(
            verbs_file, nouns_file, time_words_file, locations_file
        )
        
        # Tải trigger words theo loại sự kiện
        self.event_triggers = self._load_event_triggers(triggers_file)
        
        print(f"Đã tải {sum(len(triggers) for triggers in self.event_triggers.values())} trigger words cho {len(self.event_triggers)} loại sự kiện")
    
    def _load_event_triggers(self, triggers_file: str) -> Dict[str, List[str]]:
        """Đọc trigger words theo loại sự kiện từ file"""
        triggers = {}
        try:
            with open(triggers_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            event_type = parts[0].strip()
                            words = [w.strip() for w in parts[1].split(',')]
                            triggers[event_type] = [w for w in words if w]
        except Exception as e:
            print(f"Lỗi khi đọc file {triggers_file}: {e}")
            # Fallback values nếu không đọc được file
            triggers = {
                "Policy-Announcement": ["ban hành", "công bố", "thông báo", "khai trương", "ra mắt", 
                                      "phê duyệt", "quyết định", "thông qua", "hoàn thành", "khởi công"],
                "Leader-Activity": ["chủ trì", "làm việc", "gặp mặt", "tiếp", "hội đàm", "thăm", "phát biểu"],
                "Emergency-Event": ["cảnh báo", "khẩn cấp", "nguy hiểm", "tai nạn", "thiệt hại"]
            }
        return triggers
    
    def tokenize(self, text: str) -> List[str]:
        """Tách từ cho văn bản tiếng Việt"""
        return text.split()
    
    def detect_trigger(self, tokens: List[str], event_type: str) -> Dict:
        """Phát hiện trigger word dựa vào loại sự kiện"""
        text = " ".join(tokens).lower()
        
        # Tìm trigger từ danh sách đã định nghĩa
        triggers = self.event_triggers.get(event_type, [])
        for trigger in triggers:
            trigger_lower = trigger.lower()
            if " " in trigger_lower:
                # Xử lý từ ghép (multi-word trigger)
                if trigger_lower in text:
                    # Tìm vị trí của từ ghép trong văn bản
                    words = trigger_lower.split()
                    for i in range(len(tokens) - len(words) + 1):
                        if " ".join([t.lower() for t in tokens[i:i+len(words)]]) == trigger_lower:
                            return {
                                "text": trigger,
                                "start": i,
                                "end": i + len(words)
                            }
            else:
                # Xử lý từ đơn (single-word trigger)
                for i, token in enumerate(tokens):
                    if token.lower() == trigger_lower:
                        return {
                            "text": token,
                            "start": i,
                            "end": i + 1
                        }
        
        # Nếu không tìm thấy từ danh sách, sử dụng tokenizer để phát hiện động từ
        doc = self.tokenizer(" ".join(tokens))
        for i, token in enumerate(doc):
            if token.pos_ == "VERB" and i < len(tokens):
                # Tìm cụm động từ (verb phrase)
                verb_phrase = [tokens[i]]
                start_idx = i
                end_idx = i + 1
                
                # Mở rộng về sau để tìm cụm động từ
                j = i + 1
                while j < len(tokens) and j < len(doc) and j < i + 3:
                    if doc[j].pos_ in ["VERB", "ADV"]:
                        verb_phrase.append(tokens[j])
                        end_idx = j + 1
                    else:
                        break
                    j += 1
                
                return {
                    "text": " ".join(verb_phrase),
                    "start": start_idx,
                    "end": end_idx
                }
        
        # Fallback: lấy từ đầu tiên
        return {
            "text": tokens[0] if tokens else "",
            "start": 0,
            "end": 1 if tokens else 0
        }

    def extract_arguments(self, tokens: List[str], trigger_info: Dict) -> List[Dict]:
        """Trích xuất các arguments và vai trò của chúng"""
        arguments = []
        doc = self.tokenizer(" ".join(tokens))
        
        # Chuẩn bị danh sách các phạm vi đã được gán
        covered_spans = [(trigger_info["start"], trigger_info["end"])]
        
        # 1. Xác định Object - thường nằm sau trigger và là danh từ/cụm danh từ
        object_span = None
        object_text = []
        start_idx = None
        
        # Tìm object sau trigger
        for i in range(trigger_info["end"], len(tokens)):
            if i >= len(doc):
                continue
                
            token = doc[i]
            
            # Bắt đầu object mới
            if start_idx is None:
                if token.pos_ in ["NOUN", "X"]:
                    start_idx = i
                    object_text.append(tokens[i])
            else:
                # Tiếp tục mở rộng object
                if token.pos_ in ["NOUN", "X", "NUM"]:
                    object_text.append(tokens[i])
                else:
                    break
        
        # Nếu tìm thấy object
        if object_text and start_idx is not None:
            end_idx = start_idx + len(object_text)
            object_span = (start_idx, end_idx)
            covered_spans.append(object_span)
            arguments.append({
                "role": "Object",
                "text": " ".join(object_text),
                "start": start_idx,
                "end": end_idx
            })
        
        # 2. Xác định Agent (nếu có) - thường nằm trước trigger
        agent_span = None
        if trigger_info["start"] > 0:
            agent_text = []
            for i in range(trigger_info["start"]-1, -1, -1):
                if i >= len(doc):
                    continue
                    
                token = doc[i]
                if token.pos_ in ["NOUN", "X"]:
                    agent_text.insert(0, tokens[i])
                else:
                    break
            
            if agent_text:
                start_idx = trigger_info["start"] - len(agent_text)
                end_idx = trigger_info["start"]
                agent_span = (start_idx, end_idx)
                covered_spans.append(agent_span)
                arguments.append({
                    "role": "Agent",
                    "text": " ".join(agent_text),
                    "start": start_idx,
                    "end": end_idx
                })
        
        # 3. Xác định Location - đầu tiên kiểm tra từ chỉ vị trí như "tại", "ở"
        location_identified = False
        
        # Danh sách từ chỉ vị trí
        location_indicators = ["tại", "ở", "tại nơi", "trong", "ngoài"]
        
        # Tìm từ chỉ định vị trí trước
        for i in range(len(tokens)):
            for indicator in location_indicators:
                indicator_words = indicator.split()
                if i + len(indicator_words) <= len(tokens):
                    if " ".join([tokens[i+j].lower() for j in range(len(indicator_words))]) == indicator:
                        # Tìm thấy từ chỉ định vị trí
                        loc_start = i
                        
                        # Tìm cụm từ chỉ địa điểm sau từ chỉ định vị trí
                        j = i + len(indicator_words)
                        loc_end = j
                        
                        # Mở rộng để tìm địa điểm đầy đủ
                        while j < len(tokens):
                            if j < len(doc) and doc[j].pos_ in ["NOUN", "X", "PROPN"]:
                                loc_end = j + 1
                                j += 1
                            else:
                                break
                        
                        # Kiểm tra không trùng với những phạm vi đã gán
                        if not any(span[0] <= loc_start < span[1] or span[0] < loc_end <= span[1] for span in covered_spans):
                            loc_span = (loc_start, loc_end)
                            covered_spans.append(loc_span)
                            arguments.append({
                                "role": "Location",
                                "text": " ".join(tokens[loc_start:loc_end]),
                                "start": loc_start,
                                "end": loc_end
                            })
                            location_identified = True
                            break
            if location_identified:
                break
        
        # Nếu chưa tìm thấy location, tìm từ danh sách địa điểm
        if not location_identified:
            text_lower = " ".join(tokens).lower()
            
            # Sắp xếp locations theo độ dài giảm dần để ưu tiên tìm chuỗi dài nhất
            sorted_locations = sorted(self.tokenizer.locations, key=len, reverse=True)
            
            for location in sorted_locations:
                location_lower = location.lower()
                if location_lower in text_lower:
                    # Tìm vị trí của địa điểm trong tokens
                    loc_words = location_lower.split()
                    for i in range(len(tokens) - len(loc_words) + 1):
                        if " ".join([t.lower() for t in tokens[i:i+len(loc_words)]]) == location_lower:
                            loc_span = (i, i + len(loc_words))
                            
                            # Kiểm tra không trùng với phạm vi đã gán
                            if not any(span[0] <= loc_span[0] < span[1] or span[0] < loc_span[1] <= span[1] for span in covered_spans):
                                covered_spans.append(loc_span)
                                arguments.append({
                                    "role": "Location",
                                    "text": " ".join(tokens[i:i+len(loc_words)]),
                                    "start": i,
                                    "end": i + len(loc_words)
                                })
                                location_identified = True
                                break
                    if location_identified:
                        break
        
        # 4. Xác định Time - tìm từ có liên quan đến thời gian
        time_identified = False
        
        # Danh sách từ chỉ thời gian
        time_indicators = ["trong", "vào", "lúc", "khi", "trước", "sau", "đầu", "giữa", "cuối"]
        
        for i, token in enumerate(tokens):
            # Bỏ qua nếu token đã được xác định là location indicator
            if any(token.lower() == loc_ind for loc_ind in location_indicators):
                continue
                
            # Kiểm tra từ chỉ thời gian
            is_time_indicator = token.lower() in time_indicators
            is_time_pos = (i < len(doc) and doc[i].pos_ == "TIME")
            
            if is_time_indicator or is_time_pos:
                # Mở rộng phạm vi tìm kiếm thời gian (trước và sau)
                time_start = i
                time_end = i + 1
                
                # Mở rộng về phía trước
                j = i - 1
                while j >= 0 and j < len(doc):
                    if doc[j].pos_ == "NUM" or tokens[j].lower() in ["đầu", "giữa", "cuối", "trong"]:
                        time_start = j
                    else:
                        break
                    j -= 1
                
                # Mở rộng về phía sau
                j = i + 1
                while j < len(tokens) and j < len(doc):
                    if doc[j].pos_ in ["NUM", "TIME"] or tokens[j].lower() in ["nay", "tới", "sau", "này", "năm", "tháng", "ngày"]:
                        time_end = j + 1
                    else:
                        break
                    j += 1
                
                time_span = (time_start, time_end)
                
                # Kiểm tra không trùng với phạm vi đã gán
                if not any(span[0] <= time_span[0] < span[1] or span[0] < time_span[1] <= span[1] for span in covered_spans):
                    covered_spans.append(time_span)
                    arguments.append({
                        "role": "Time",
                        "text": " ".join(tokens[time_start:time_end]),
                        "start": time_start,
                        "end": time_end
                    })
                    time_identified = True
                    break
        
        return arguments
    
    def process_document(self, doc: Dict[str, str]) -> Dict[str, Any]:
        """Xử lý một tài liệu và thêm thông tin trigger và arguments"""
        content = doc["content"]
        event_type = doc["event_type"]
        doc_id = doc["doc_id"]
        
        # Tokenize nội dung
        tokens = self.tokenize(content)
        
        # Phát hiện trigger
        trigger_info = self.detect_trigger(tokens, event_type)
        
        # Trích xuất arguments
        arguments = self.extract_arguments(tokens, trigger_info)
        
        # Tạo kết quả
        result = {
            "doc_id": doc_id,
            "sent_id": doc_id,
            "tokens": tokens,
            "event_mentions": [
                {
                    "id": f"event-{doc_id.split('-')[1]}",
                    "event_type": event_type,
                    "trigger": trigger_info,
                    "arguments": arguments
                }
            ]
        }
        
        return result

def process_dataset(input_file: str, output_file: str, 
                   verbs_file: str, nouns_file: str, 
                   time_words_file: str, locations_file: str,
                   triggers_file: str):
    """
    Xử lý toàn bộ tập dữ liệu từ file JSON đầu vào
    
    Args:
        input_file: File JSON chứa dữ liệu đầu vào
        output_file: File JSON để lưu kết quả
        verbs_file: File chứa danh sách động từ
        nouns_file: File chứa danh sách danh từ
        time_words_file: File chứa từ thời gian
        locations_file: File chứa địa điểm
        triggers_file: File chứa trigger words theo loại sự kiện
    """
    # Kiểm tra tồn tại các file từ điển
    file_paths = [verbs_file, nouns_file, time_words_file, locations_file, triggers_file]
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file từ điển {file_path}")
            return
    
    tagger = VietnameseEventTagger(
        verbs_file, nouns_file, time_words_file, locations_file, triggers_file
    )
    processed_docs = []
    
    # Đọc dữ liệu đầu vào
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            raw_docs = json.load(f)
            # Kiểm tra nếu raw_docs không phải một list (có thể là dictionary đơn lẻ)
            if not isinstance(raw_docs, list):
                raw_docs = [raw_docs]
        except json.JSONDecodeError:
            print(f"Lỗi: File {input_file} không phải định dạng JSON hợp lệ.")
            return
    
    # Xử lý từng tài liệu
    print(f"Đang xử lý {len(raw_docs)} tài liệu...")
    for doc in raw_docs:
        processed = tagger.process_document(doc)
        processed_docs.append(processed)
    
    # Lưu kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)
    
    print(f"Đã xử lý {len(processed_docs)} tài liệu và lưu vào {output_file}")
    
    # In ra ví dụ kết quả đầu tiên để kiểm tra
    if processed_docs:
        print("\nVí dụ kết quả đầu tiên:")
        print(json.dumps(processed_docs[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # Đường dẫn đến các file từ điển
    verbs_file = "verbs.txt"       # File chứa danh sách động từ
    nouns_file = "nouns.txt"       # File chứa danh sách danh từ
    time_file = "time_words.txt"   # File chứa từ thời gian
    loc_file = "locations.txt"     # File chứa địa điểm
    triggers_file = "event_triggers.txt"  # File chứa trigger words
    
    # File dữ liệu đầu vào và đầu ra
    input_file = "data1.json"
    output_file = "tagged_events.json"
    
    # Kiểm tra tồn tại file đầu vào
    if not os.path.exists(input_file):
        print(f"Lỗi: File {input_file} không tồn tại.")
        print("Tạo file mẫu để kiểm thử...")
        
        # Đảm bảo thư mục đầu vào tồn tại
        input_dir = os.path.dirname(input_file)
        if input_dir and not os.path.exists(input_dir):
            os.makedirs(input_dir)
        
        # Tạo một file mẫu để thử nghiệm
        sample_data = [
            {
                "doc_id": "train-00001",
                "content": "hoàn thành cao tốc quảng ngãi bình định trong năm nay",
                "date": "09-02-2025",
                "event_type": "Policy-Announcement"
            },
            {
                "doc_id": "train-00002",
                "content": "thủ tướng chủ trì họp về tình hình kinh tế xã hội",
                "date": "10-02-2025",
                "event_type": "Leader-Activity"
            }
        ]
        
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"Đã tạo file mẫu {input_file}")
    
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tiến hành xử lý
    process_dataset(input_file, output_file, 
                   verbs_file, nouns_file, time_file, loc_file, triggers_file)