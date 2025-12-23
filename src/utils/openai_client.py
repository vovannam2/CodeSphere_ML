"""
Client để gọi OpenAI API
Sử dụng cùng config như Java backend
"""

import os
import requests
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    """Client để gọi OpenAI API"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        self.api_url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1/chat/completions')
        self.model = os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '4096'))
    
    def call_api(self, system_prompt: str, user_message: str) -> Optional[str]:
        """
        Gọi OpenAI API
        
        Parameters:
        - system_prompt: System prompt
        - user_message: User message
        
        Returns:
        - Response text hoặc None nếu lỗi
        """
        if not self.api_key:
            print("Cảnh báo: OpenAI API key chưa được cấu hình")
            return None
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            messages = []
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            messages.append({
                'role': 'user',
                'content': user_message
            })
            
            payload = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            # Parse response
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                return content
            
            return None
            
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API: {e}")
            return None
    
    def _explain_predicted_rating(self) -> str:
        """
        ⭐ GIẢI THÍCH: PREDICTED_RATING LÀ GÌ?
        =======================================
        
        PREDICTED_RATING (điểm dự đoán) của mỗi bài là kết quả của ML Model
        được train từ DỮ LIỆU LỊCH SỬ của toàn bộ user trên hệ thống.
        
        🔑 CÔNG THỨC TÍNH (Collaborative Filtering + Content-Based):
        ============================================================
        
        Ví dụ: Tính predicted_rating cho user_5 với problem_999 (user_5 chưa từng làm)
        
        Input dữ liệu:
        - user_5: solved=[1,2,3,4,5], attempted=[6,7], bookmarked=[10,20]
        - Toàn bộ user trên hệ thống: user_1, user_2, ..., user_100
        - Tất cả problem: problem_1, problem_2, ..., problem_10000
        
        BƯỚC 1: COLLABORATIVE FILTERING (User-to-User)
        ===============================================
        Tìm những user GIỐNG user_5
        
        Ví dụ:
        - user_5: solved=[1,2,3,4,5] (giỏi Array)
        - user_15: solved=[1,2,3,4,5,100,101,102] (cũng giỏi Array)
        - user_42: solved=[200,201,202,203] (giỏi String)
        
        → user_5 tương tự user_15 (cùng giỏi Array)
        → Nếu user_15 solve được problem_999 dễ dàng
        → Dự đoán user_5 cũng solve được problem_999 dễ dàng
        → predicted_rating = cao (0.8+)
        
        BƯỚC 2: CONTENT-BASED FILTERING (Problem-to-Problem)
        ====================================================
        Tìm những bài TƯƠNG TỰ bài user_5 đã giỏi
        
        Ví dụ:
        - user_5 giỏi: Two Sum (Array, Easy), Best Time to Buy Stock (Array, Medium)
        - problem_999 là: Contains Duplicate (Array, Easy) - GIỐNG loại bài user_5 giỏi
        - problem_998 là: Reverse Linked List (LinkedList, Medium) - KHÁC loại bài
        
        → problem_999 (Array, Easy) tương tự bài user_5 giỏi
        → predicted_rating = cao (0.8+)
        
        BƯỚC 3: FEATURE ENGINEERING (Tính năng của bài)
        ===============================================
        - Difficulty: Easy (user_5 giỏi Easy bài)
        - Category: Array (user_5 giỏi Array)
        - Tags: [sorting, searching] (giống bài user_5 làm)
        - Acceptance Rate: 50% (bài này khó trung bình)
        - Topic Relatedness: 0.9 (bài này liên quan đến bài user_5 làm)
        
        → Kết hợp các feature này → predicted_rating
        
        BƯỚC 4: RATING SCORE (Combine các phương pháp)
        =============================================
        predicted_rating = (
            0.4 * collaborative_score +    # 40% dựa trên user tương tự
            0.3 * content_score +          # 30% dựa trên bài tương tự
            0.2 * difficulty_score +       # 20% dựa trên độ khó phù hợp
            0.1 * popularity_score         # 10% dựa trên độ phổ biến
        )
        
        Ví dụ:
        - collaborative_score = 0.85 (user tương tự solve được)
        - content_score = 0.80 (bài tương tự bài user giỏi)
        - difficulty_score = 0.90 (độ khó phù hợp user level)
        - popularity_score = 0.75 (bài khá phổ biến)
        
        → predicted_rating = 0.4*0.85 + 0.3*0.80 + 0.2*0.90 + 0.1*0.75
                          = 0.34 + 0.24 + 0.18 + 0.075
                          = 0.835 ✅
        
        ====================================================
        THỰC TIỄN: MÔ HÌNH ML ĐƯỢC HỌC TỪ ĐÂU?
        ====================================================
        
        Training Data (từ lịch sử tất cả user):
        
        User 1: {solved_ids=[1,2,3], attempted_ids=[4,5]}
        User 2: {solved_ids=[1,2,6,7], attempted_ids=[8]}
        User 3: {solved_ids=[10,20,30], attempted_ids=[5]}
        ...
        User 100: {solved_ids=[...], attempted_ids=[...]}
        
        Problem Features:
        problem_1: {category='Array', difficulty='Easy', tags=[...], ...}
        problem_2: {category='String', difficulty='Easy', tags=[...], ...}
        ...
        
        ML Model học từ patterns:
        - Người dùng nào giỏi category gì?
        - Người dùng nào thích difficulty nào?
        - Bài nào user thường solve nếu đã solve bài khác?
        - Success rate của người giỏi vs người yếu
        
        → Sau khi train → Model có thể dự đoán:
          "Nếu user_new có profile giống user_1"
          → "user_new sẽ giỏi problem_X với xác suất 0.85"
        
        ====================================================
        TRƯỜNG HỢP THỰC TẾ: VÍ DỤ CỤ THỂ
        ====================================================
        
        User A: solved=100 bài, attempted=20 bài
        Success rate = 100 / (100+20) = 83% → Advanced level
        
        problem_999 chưa từng làm:
        - category: Array
        - difficulty: Hard
        - predicted_rating: 0.78
        
        Lý do dự đoán 0.78:
        ✅ User A giỏi Array (solved 60 bài Array) → +0.3
        ✅ User A giỏi Hard level (solved 40 bài Hard) → +0.3
        ✅ Người tương tự user A (Advanced level) thường solve được bài này → +0.2
        ✅ Bài này khá phổ biến và được many users solve → +0.1
        → Tổng = 0.7-0.8 (điểm dự đoán khá cao)
        
        Nhưng nếu:
        - category: DP
        - difficulty: Very Hard
        - predicted_rating: 0.45
        
        Lý do dự đoán 0.45:
        ❌ User A chỉ solved 5 bài DP → -0.2
        ❌ User A chưa từng solve Very Hard → -0.2
        ❌ Bài này khó, user tương tự cũng không solve được → -0.1
        → Tổng = 0.4-0.5 (điểm dự đoán thấp → không recommend)
        
        ====================================================
        KẾT LUẬN
        ====================================================
        
        predicted_rating KHÔNG được tính từ:
        ❌ Lịch sử của user này (vì user chưa làm bài này)
        
        predicted_rating ĐƯỢC tính từ:
        ✅ ML Model train từ lịch sử của ALL users
        ✅ Profile của user này (category giỏi, level, success rate)
        ✅ Features của bài (difficulty, category, tags)
        ✅ Pattern: "người giống user này thường giỏi loại bài này"
        
        Đó là lý do tại sao ta có thể recommend bài chưa từng làm!
        """
        return ""
    
    def _fallback_recommendations(self, top_problems: list, user_history: dict = None) -> dict:
        """
        Fallback function: Trả về top problems từ local model với intelligent filtering
        
        ⭐ QUAN TRỌNG: 
        ==============
        DỮ LIỆU SỬ DỤNG LÀ CỦA USER (NGƯỜI DÙNG HỆ THỐNG)
        KHÔNG PHẢI CỦA BẠN (DEVELOPER)
        
        Ví dụ:
        - solved=5: Người dùng đã giải quyết 5 bài tập
        - attempted=2: Người dùng đã cố gắng 2 bài nhưng chưa xong
        - bookmarked=3: Người dùng đã bookmark 3 bài
        
        Mỗi người dùng có dữ liệu khác nhau!
        User A: (solved=100, attempted=20, bookmarked=10)
        User B: (solved=5, attempted=2, bookmarked=3)
        → Recommend khác nhau cho mỗi người
        """
        print(f"[Fallback] Using intelligent local model recommendations")
        
        if user_history is None:
            user_history = {}
        
        # ============ BƯỚC 1: FILTER DỮ LIỆU ============
        # DÙNG: Dữ liệu của NGƯỜI DÙNG hiện tại
        # Lấy ID của các bài mà NGƯỜI DÙNG này đã solved
        user_solved_ids = set(user_history.get('solved', []))
        # Lấy ID của các bài mà NGƯỜI DÙNG này đã attempted
        user_attempted_ids = set(user_history.get('attempted', []))
        
        # Ví dụ:
        # User A: user_solved_ids = {1, 2, 3, 4, 5}
        # User B: user_solved_ids = {10, 20, 30}
        
        # Chỉ giữ lại bài chưa làm qua (của NGƯỜI DÙNG này)
        filtered_problems = [
            p for p in top_problems 
            if p['problem_id'] not in user_solved_ids and p['problem_id'] not in user_attempted_ids
        ]
        
        print(f"[Fallback] Filtered from {len(top_problems)} to {len(filtered_problems)} (removed solved/attempted)")
        
        # Fallback nếu tất cả bài đều làm rồi
        if not filtered_problems:
            print(f"[Fallback] No new problems available, returning top unsolved")
            filtered_problems = top_problems[:10]
        
        # ============ BƯỚC 2: PHÂN TÍCH LEVEL NGƯỜI DÙNG ============
        # DÙNG: Dữ liệu lịch sử của NGƯỜI DÙNG này
        # Tính tỷ lệ thành công = solved / (solved + attempted)
        # Ví dụ:
        print(f"[Fallback] Filtered from {len(top_problems)} to {len(filtered_problems)} (removed solved/attempted)")
        
        # Fallback nếu tất cả bài đều làm rồi
        if not filtered_problems:
            print(f"[Fallback] No new problems available, returning top unsolved")
            filtered_problems = top_problems[:10]
        
        # ============ BƯỚC 2: PHÂN TÍCH LEVEL NGƯỜI DÙNG ============
        # Tính tỷ lệ thành công
        total_attempts = len(user_solved_ids) + len(user_attempted_ids)
        success_rate = len(user_solved_ids) / max(total_attempts, 1)
        
        # Phân loại level dựa trên success rate
        if success_rate >= 0.8:
            # Người dùng giỏi: ưu tiên Hard, Medium, Easy (thử thách cao)
            user_level = "Advanced"
            difficulty_preference = ["Hard", "Medium", "Easy"]
        elif success_rate >= 0.5:
            # Người dùng trung bình: ưu tiên Medium, Hard, Easy (cân bằng)
            user_level = "Intermediate"
            difficulty_preference = ["Medium", "Hard", "Easy"]
        else:
            # Người dùng mới: ưu tiên Easy, Medium, Hard (xây dựng nền tảng)
            user_level = "Beginner"
            difficulty_preference = ["Easy", "Medium", "Hard"]
        
        print(f"[Fallback] User level: {user_level} (Success rate: {success_rate:.1%})")
        
        # ============ BƯỚC 3: NHÓM BÀI THEO CATEGORY ============
        # Ví dụ: {
        #   "Array": [problem1, problem2, ...],
        #   "String": [problem3, problem4, ...],
        #   "DP": [problem5, ...]
        # }
        category_groups = {}
        for problem in filtered_problems:
            category = problem.get('category', 'Other')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(problem)
        
        print(f"[Fallback] Problems grouped into {len(category_groups)} categories")
        
        # ============ BƯỚC 4: CHỌN BÀI ĐA DẠNG ============
        selected = []
        
        # Sort bài trong mỗi category theo score (cao → thấp)
        # Mục đích: Lấy bài tốt nhất từ mỗi category
        for category in category_groups:
            category_groups[category].sort(
                key=lambda x: x['predicted_rating'],
                reverse=True
            )
        
        # Lấy tối đa 2 bài từ mỗi category (round-robin)
        # Ví dụ: Array[0], String[0], DP[0], Array[1], String[1], DP[1], ...
        max_per_category = 2
        for category in sorted(category_groups.keys()):
            for problem in category_groups[category][:max_per_category]:
                if len(selected) < 10:  # Chỉ cần 10 bài
                    selected.append(problem)
        
        # Nếu chưa đủ 10 bài, lấy những bài score cao nhất từ các category khác
        all_remaining = [p for p in filtered_problems if p not in selected]
        all_remaining.sort(key=lambda x: x['predicted_rating'], reverse=True)
        for problem in all_remaining:
            if len(selected) < 10:
                selected.append(problem)
        
        print(f"[Fallback] Selected {len(selected)} problems with diversity")
        
        # ============ BƯỚC 5: SINH EXPLANATION ============
        fallback_recommendations = []
        current_categories = set()
        
        for idx, problem in enumerate(selected):
            category = problem.get('category', 'Other')
            difficulty = problem.get('difficulty', 'N/A')
            
            # Gọi hàm để sinh lý do recommend
            reason = self._generate_fallback_reason(
                problem,
                user_level,
                idx,
                category,
                current_categories
            )
            
            current_categories.add(category)  # Đánh dấu đã dùng category này
            
            fallback_recommendations.append({
                "problem_id": problem['problem_id'],
                "title": problem.get('title', 'N/A'),
                "difficulty": difficulty,
                "reason": reason,  # Lý do recommend
                "skills_learned": category,
                "learning_path": f"Step {idx + 1} - Build {category} skills"
            })
        
        return {
            "analysis": f"Recommended for {user_level} level programmer. Success rate: {success_rate:.1%}. Mix of {len(category_groups)} categories to build diverse skills.",
            "recommendations": fallback_recommendations,
            "source": "local_intelligent"
        }
    
    def _generate_fallback_reason(self, problem: dict, user_level: str, position: int, category: str, used_categories: set) -> str:
        """
        CRITERIA FOR GENERATING RECOMMENDATION REASONS
        ===============================================
        
        4 combined criteria:
        1. SCORE-BASED: How confident is the AI model?
           - 0.8+: Very good match
           - 0.6-0.8: Moderate match
           - <0.6: To learn new skills
        
        2. POSITION-BASED: Position in learning path?
           - Position 0: Start here (first problem)
           - Position 1-2: Foundational
           - Position 3+: Progressive challenge
        
        3. CATEGORY-BASED: Old or new category?
           - If new category: "Introduce new category: X"
           - If old category: "Deepen X skills"
        
        4. DIFFICULTY-BASED: Difficulty matches user level?
           - Beginner + Easy: "Perfect starting point"
           - Intermediate + Medium: "Matches your level"
           - Advanced + Hard: "Push your limits"
        """
        score = problem.get('predicted_rating', 0)
        difficulty = problem.get('difficulty', 'N/A')
        
        reasons = []
        
        # CRITERIA 1: SCORE-BASED
        if score >= 0.8:
            reasons.append("AI model predicts high match (Score: {:.2f})".format(score))
        elif score >= 0.6:
            reasons.append("Good match based on learning pattern (Score: {:.2f})".format(score))
        else:
            reasons.append("Recommended to build new skills (Score: {:.2f})".format(score))
        
        # CRITERIA 2: POSITION-BASED
        if position == 0:
            reasons.append("Start here - best match for your current level")
        elif position < 3:
            reasons.append("Early sequence - foundational for next steps")
        else:
            reasons.append("Progressive challenge to expand knowledge")
        
        # CRITERIA 3: CATEGORY-BASED
        if category not in used_categories:
            reasons.append("Introduce new category: {}".format(category))
        else:
            reasons.append("Deepen {} skills".format(category))
        
        # CRITERIA 4: DIFFICULTY-BASED
        if user_level == "Beginner" and difficulty == "Easy":
            reasons.append("Perfect starting point - not overwhelming")
        elif user_level == "Intermediate" and difficulty == "Medium":
            reasons.append("Matches your current challenge level")
        elif user_level == "Advanced" and difficulty == "Hard":
            reasons.append("Push your limits to master advanced concepts")
        
        return " | ".join(reasons)
    
    def refine_recommendations(
        self, 
        top_problems: list, 
        user_history: dict,
        user_id: int
    ) -> Optional[dict]:
        """
        Dùng OpenAI để refine recommendations và thêm explanation chi tiết
        
        Parameters:
        - top_problems: List các problems từ Local Model (top 20)
        - user_history: Lịch sử của user (solved, attempted, bookmarked)
        - user_id: ID của user
        
        Returns:
        - Dict với refined recommendations và detailed explanations
        """
        # Validate input
        if not top_problems or len(top_problems) == 0:
            print(f"Warning: Empty top_problems list for user {user_id}")
            return {
                "analysis": "No recommendations from local model",
                "recommendations": [],
                "source": "empty_input"
            }
        
        print(f"[Refine] Starting refinement for user {user_id} with {len(top_problems)} problems")
        
        # Tạo prompt chi tiết hơn
        problems_text = "\n".join([
            f"- Problem #{p['problem_id']}: {p.get('title', 'N/A')} "
            f"(Difficulty: {p.get('difficulty', 'N/A')}, "
            f"Category: {p.get('category', 'N/A')}, "
            f"AI Score: {p['predicted_rating']:.2f})"
            for p in top_problems[:20]
        ])
        
        user_solved = user_history.get('solved', [])
        user_attempted = user_history.get('attempted', [])
        user_bookmarked = user_history.get('bookmarked', [])
        
        print(f"[Refine] User stats - Solved: {len(user_solved)}, Attempted: {len(user_attempted)}, Bookmarked: {len(user_bookmarked)}")
        
        system_prompt = """You are an experienced programming learning advisor.
Your task is to:
1. Analyze the user's profile (experience, learning history)
2. Select the top 10 most suitable problems from the list of 20 recommended problems
3. Provide DETAILED explanations in English for why each problem is suitable:
   - Connection to the user's current experience
   - Skills that will be improved
   - Why the difficulty level is appropriate
   - How this problem helps the user develop
4. Sort by optimal learning order (easy → hard)

IMPORTANT: All explanations must be in English.

Return in exact JSON format:
{
  "analysis": "Brief analysis of user profile...",
  "recommendations": [
    {
      "problem_id": 123,
      "title": "Problem name",
      "difficulty": "Medium",
      "reason": "Detailed reason in English...",
      "skills_learned": "Skills to be learned...",
      "learning_path": "Position in learning path..."
    }
  ]
}

IMPORTANT: Must return at least 5-10 recommendations, do not leave recommendations empty!"""
        
        user_message = f"""User ID: {user_id}

Learning History:
- Solved: {len(user_solved)} problems
- Attempted: {len(user_attempted)} problems
- Bookmarked: {len(user_bookmarked)} problems
- Success rate: {len(user_solved) / max(len(user_solved) + len(user_attempted), 1) * 100:.1f}%

List of 20 problems recommended by AI model (sorted by score):
{problems_text}

Please analyze the user profile and select the top 10 most suitable problems with:
- DETAILED explanation in English for why each is recommended
- Skills that will be learned
- Position in the user's learning path
- Sort by logical learning order

IMPORTANT: Return at least 5-10 recommendations, do not leave empty!
Return in JSON format as requested.
All explanations must be in English."""
        
        response = self.call_api(system_prompt, user_message)
        
        if not response:
            print(f"[Refine] Error: No response from OpenAI API")
            return self._fallback_recommendations(top_problems, user_history)
        
        print(f"[Refine] OpenAI Response length: {len(response)} characters")
        
        try:
            # Parse JSON từ response
            json_str = response.strip()
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                print(f"[Refine] Extracted JSON from markdown code block")
            elif '```' in response:
                json_start = response.find('```') + 3
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                print(f"[Refine] Extracted JSON from code block")
            
            result = json.loads(json_str)
            
            # Validate result structure
            if 'recommendations' not in result:
                print("[Refine] Error: missing 'recommendations' field")
                return self._fallback_recommendations(top_problems, user_history)
            
            if not isinstance(result['recommendations'], list):
                print("[Refine] Error: 'recommendations' is not a list")
                return self._fallback_recommendations(top_problems, user_history)
            
            if len(result['recommendations']) == 0:
                print("[Refine] Warning: Empty recommendations list from OpenAI, using fallback")
                return self._fallback_recommendations(top_problems, user_history)
            
            print(f"[Refine] Successfully refined {len(result['recommendations'])} recommendations")
            result['source'] = 'openai'
            return result
            
        except json.JSONDecodeError as e:
            print(f"[Refine] JSON Parse error: {e}")
            print(f"[Refine] Response: {response[:300]}...")
            return self._fallback_recommendations(top_problems, user_history)
        except Exception as e:
            print(f"[Refine] Error: {e}")
            return self._fallback_recommendations(top_problems, user_history)

