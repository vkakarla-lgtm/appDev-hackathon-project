from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from bson import ObjectId
from pymongo import MongoClient
import bcrypt
import jwt
import os
import random
import string
import math

# --- Configuration & Initialization ---

app = FastAPI(title="Fantasy Life League API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "fantasy-life-league-secret-key-2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# MongoDB Connection (Using simple local placeholder for a self-contained environment)
# In a real hackathon deployment, MONGO_URI would point to MongoDB Atlas.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["fantasy_life_league"]

# Collections
users_collection = db["users"]
leagues_collection = db["leagues"]
matchups_collection = db["matchups"]
chats_collection = db["league_chat"]
habit_entries_collection = db["habit_entries"]

# Security
security = HTTPBearer()

# --- Utility Functions ---

def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to convert ObjectId to string and remove sensitive fields."""
    if doc:
        doc["_id"] = str(doc["_id"])
        if "password_hash" in doc:
            del doc["password_hash"]
    return doc

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Generates JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verifies JWT token and returns user ID."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

def get_current_user_doc(user_id: str) -> Dict[str, Any]:
    """Fetches user document from DB."""
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

# --- Data Models (Pydantic) ---

class HabitGoal(BaseModel):
    sleep: Optional[float] = 8.0 # Hours
    study: Optional[float] = 2.0 # Hours
    exercise: Optional[float] = 1.0 # Hours
    hydration: Optional[int] = 8 # Cups
    nutrition: Optional[int] = 1 # Healthy meals (1 or 0)

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserIn(UserBase):
    password: str

class UserDB(UserBase):
    # Python field name is `id`, but it maps to Mongo's `_id`
    id: str = Field(alias="_id")
    goals: HabitGoal = HabitGoal()
    league_id: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class HabitEntryIn(BaseModel):
    sleep: Optional[float] = None
    study: Optional[float] = None
    exercise: Optional[float] = None
    hydration: Optional[int] = None
    nutrition: Optional[int] = None
    mindfulness: Optional[int] = None # Minutes

class HabitEntryOut(HabitEntryIn):
    date: str
    points: Dict[str, float]
    total_points: float

class LeagueIn(BaseModel):
    name: str

class LeagueJoin(BaseModel):
    code: str

class LeagueChatMessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class LeagueChatMessageOut(BaseModel):
    id: str = Field(alias="_id")
    league_id: str
    user_id: str
    user_name: str
    message: str
    timestamp: str  # ISO 8601 string

# --- Core Logic: Scoring, Streaks, Education (Educate/Track/Reward) ---

MAX_POINTS_PER_CATEGORY = 10.0

def calculate_score(entry: Dict[str, Any], goals: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates points for each habit category based on logged value vs. goal.
    Points are capped at MAX_POINTS_PER_CATEGORY (10.0).
    """
    points = {}
    
    for category, value in entry.items():
        if category in goals and value is not None:
            goal = goals.get(category)
            if goal and goal > 0:
                # Calculate ratio (capped at 1.5x goal to reward over-achieving slightly)
                ratio = min(value / goal, 1.5)
                points[category] = round(ratio * MAX_POINTS_PER_CATEGORY, 1)
            elif category == "nutrition" and value is not None:
                # Binary goal (1 means healthy meal logged)
                points[category] = MAX_POINTS_PER_CATEGORY if value >= 1 else 0.0
            else:
                # Default to a base score if no goal or value is logged
                points[category] = 0.0
        else:
            points[category] = 0.0
            
    return points

COACH_TIPS = {
    "sleep": [
        ("low", "You slept less than 6 hours. Getting 7-9 hours consistently can significantly boost your cognitive function and mood."),
        ("medium", "7 hours of sleep is solid! Aim for 8 hours tonight to ensure peak restorative brain function."),
        ("high", "8+ hours of rest is a huge win! This consistency is key to muscle recovery and memory consolidation."),
    ],
    "study": [
        ("low", "Less than 1 hour of focused work. Try the Pomodoro technique to break down tasks into manageable, focused chunks."),
        ("medium", "2 hours of focus is productive. Remember to take a 5-10 minute break every hour to avoid burnout."),
        ("high", "You crushed your study goal! Celebrate the win, and ensure you get adequate sleep to solidify that learning."),
    ],
    "exercise": [
        ("low", "No exercise today. Even a 15-minute walk can release endorphins and improve cardiovascular health."),
        ("medium", "You hit your goal! Consistent, moderate activity is more beneficial than sporadic, intense sessions."),
        ("high", "Great job pushing your limit! Don't forget proper cool-down and hydration for muscle repair."),
    ],
    "hydration": [
        ("low", "Hydration is low. Dehydration can cause fatigue and headaches. Keep a water bottle visible to prompt continuous sipping."),
        ("high", "Goal achieved! Proper hydration supports metabolism and energy levels throughout the day."),
    ],
    "general": [
        "Consistency is the biggest lever for health. Try to log at least one habit every day this week!",
        "Did you know logging your habits makes you 42% more likely to achieve them? Keep tracking!",
        "Small steps lead to big wins. Don't worry about perfection, focus on effort today."
    ]
}

def get_coach_tip(entry: Dict[str, Any], goals: Dict[str, Any]) -> Dict[str, str]:
    """Generates a context-aware coach tip based on today's logged data."""
    
    # Priority check: Sleep is the most important
    if 'sleep' in entry and goals.get('sleep', 0) > 0:
        sleep_ratio = entry['sleep'] / goals['sleep']
        if entry['sleep'] < 6:
            return {"category": "sleep", "tip": COACH_TIPS["sleep"][0][1]}
        elif sleep_ratio >= 1:
             return {"category": "sleep", "tip": COACH_TIPS["sleep"][2][1]}
        else:
             return {"category": "sleep", "tip": COACH_TIPS["sleep"][1][1]}

    # Secondary check: Hydration
    if 'hydration' in entry and goals.get('hydration', 0) > 0 and entry['hydration'] >= goals['hydration']:
        return {"category": "hydration", "tip": COACH_TIPS["hydration"][1][1]}
    
    # Default to general advice if no high-priority habits were logged or they were moderate
    return {"category": "general", "tip": random.choice(COACH_TIPS["general"])}

def calculate_streak(user_id: str) -> int:
    """Calculates the current streak of consecutive days with at least one logged habit."""
    
    entries = list(habit_entries_collection.find(
        {"user_id": ObjectId(user_id)},
        {"date": 1}
    ).sort("date", -1)) # Sort descending by date
    
    if not entries:
        return 0
    
    logged_dates = {datetime.strptime(e["date"], "%Y-%m-%d").date() for e in entries}
    
    streak = 0
    current_date = datetime.now().date()
    
    # Check for today's log first, then yesterday, and so on.
    while current_date in logged_dates or current_date == datetime.now().date():
        if current_date in logged_dates:
            streak += 1
        
        # Move to the previous day
        current_date -= timedelta(days=1)
        
        # Stop checking if we go too far back without a log
        if streak > 0 and current_date not in logged_dates:
            break
        if streak == 0 and (datetime.now().date() - current_date).days > 2:
             break # If no log today or yesterday, streak is 0

    return streak


# --- Auth Endpoints ---

@app.post("/auth/signup", response_model=Token)
async def register_user(user_in: UserIn):
    """Registers a new user and returns a JWT token."""
    if users_collection.find_one({"email": user_in.email}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed_password = bcrypt.hashpw(user_in.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    new_user = {
        "name": user_in.name,
        "email": user_in.email,
        "password_hash": hashed_password,
        "goals": HabitGoal().dict(),
        "league_id": None,
        "created_at": datetime.utcnow()
    }
    result = users_collection.insert_one(new_user)
    
    access_token = create_access_token(data={"user_id": str(result.inserted_id)})
    return {"access_token": access_token}

@app.post("/auth/login", response_model=Token)
async def login_for_access_token(user_in: UserIn):
    """Authenticates a user and returns a JWT token."""
    user = users_collection.find_one({"email": user_in.email})
    
    if not user or not bcrypt.checkpw(user_in.password.encode('utf-8'), user["password_hash"].encode('utf-8')):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"user_id": str(user["_id"])})
    return {"access_token": access_token}

@app.get("/auth/me", response_model=UserDB)
async def read_users_me(user_id: str = Depends(verify_token)):
    """Retrieves the current authenticated user's profile and goals."""
    user_doc = get_current_user_doc(user_id)
    return serialize_doc(user_doc)

# --- Habit Entry Endpoints (Track/Educate) ---

@app.post("/habits/log", response_model=HabitEntryOut)
async def log_habit_entry(
    entry_in: HabitEntryIn, 
    user_id: str = Depends(verify_token)
):
    """Logs a new habit entry for the current day, calculates points, and returns the entry."""
    user = get_current_user_doc(user_id)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Calculate Score
    points = calculate_score(entry_in.dict(exclude_none=True), user.get("goals", {}))
    total_points = sum(points.values())
    
    # 2. Prepare/Update Entry
    entry_data = entry_in.dict(exclude_none=True)
    
    new_entry = {
        "user_id": ObjectId(user_id),
        "date": today,
        "entry": entry_data,
        "points": points,
        "total_points": total_points,
        "logged_at": datetime.utcnow()
    }
    
    # Use upsert (update or insert) based on date/user_id
    result = habit_entries_collection.update_one(
        {"user_id": ObjectId(user_id), "date": today},
        {"$set": new_entry},
        upsert=True
    )
    
    # 3. Fetch the saved/updated entry
    saved_entry = habit_entries_collection.find_one({"user_id": ObjectId(user_id), "date": today})

    return {
        **saved_entry["entry"],
        "date": saved_entry["date"],
        "points": saved_entry["points"],
        "total_points": saved_entry["total_points"]
    }

@app.get("/habits/today", response_model=HabitEntryOut)
async def get_today_entry(user_id: str = Depends(verify_token)):
    """Retrieves today's habit entry."""
    today = datetime.now().strftime("%Y-%m-%d")
    entry = habit_entries_collection.find_one(
        {"user_id": ObjectId(user_id), "date": today}
    )
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No entry logged today")

    return {
        **entry["entry"],
        "date": entry["date"],
        "points": entry["points"],
        "total_points": entry["total_points"]
    }

@app.get("/habits/week")
async def get_weekly_summary(user_id: str = Depends(verify_token)):
    """Retrieves total score and category breakdown for the current week."""
    
    # Determine the current week start (Monday)
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday())
    start_date_str = start_of_week.strftime("%Y-%m-%d")
    
    # Fetch all entries since the start of the week
    entries = list(habit_entries_collection.find({
        "user_id": ObjectId(user_id),
        "date": {"$gte": start_date_str}
    }))
    
    if not entries:
        return {"week": today.isocalendar()[1], "score": {"days_logged": 0, "total": 0.0, "categories": {}}}

    total_points = 0.0
    category_totals = {}
    days_logged = len(entries)
    
    for entry in entries:
        total_points += entry["total_points"]
        for cat, pts in entry["points"].items():
            category_totals[cat] = category_totals.get(cat, 0.0) + pts

    return {
        "week": today.isocalendar()[1],
        "score": {
            "days_logged": days_logged,
            "total": round(total_points, 1),
            "categories": {k: round(v, 1) for k, v in category_totals.items()}
        }
    }

@app.get("/habits/history/{days}")
async def get_habit_history(days: int = Query(30, ge=7, le=90), user_id: str = Depends(verify_token)):
    """Retrieves historical data for charting purposes."""
    
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    entries = list(habit_entries_collection.find({
        "user_id": ObjectId(user_id),
        "date": {"$gte": start_date}
    }).sort("date", 1)) # Sort ascending
    
    history_data = []
    
    # Generate a full date range to ensure the chart has continuous x-axis labels
    date_cursor = datetime.strptime(start_date, "%Y-%m-%d").date()
    today_date = datetime.now().date()

    entry_map = {e['date']: e for e in entries}

    while date_cursor <= today_date:
        date_str = date_cursor.strftime("%Y-%m-%d")
        data_point = {"date": date_str}
        
        if date_str in entry_map:
            entry = entry_map[date_str]
            data_point["sleep"] = entry["entry"].get("sleep", 0)
            data_point["study"] = entry["entry"].get("study", 0)
            data_point["total_points"] = entry["total_points"]
        else:
            data_point["sleep"] = 0
            data_point["study"] = 0
            data_point["total_points"] = 0

        history_data.append(data_point)
        date_cursor += timedelta(days=1)
        
    return history_data


# --- League Endpoints (Reward) ---

def generate_league_code():
    """Generates a random 6-character uppercase league code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

@app.post("/league/create")
async def create_league(league_in: LeagueIn, user_id: str = Depends(verify_token)):
    """Creates a new league and adds the user as the first member."""
    user = get_current_user_doc(user_id)
    if user.get("league_id"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is already in a league")
    
    new_code = generate_league_code()
    
    new_league = {
        "name": league_in.name,
        "code": new_code,
        "members": [ObjectId(user_id)],
        "created_by": ObjectId(user_id),
        "created_at": datetime.utcnow(),
        "status": "SETUP", # SETUP, ACTIVE, COMPLETED
    }
    result = leagues_collection.insert_one(new_league)
    league_id = str(result.inserted_id)
    
    # Update user's league_id
    users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"league_id": league_id}})
    
    return {"id": league_id, "name": league_in.name, "code": new_code}

@app.post("/league/join")
async def join_league(join_in: LeagueJoin, user_id: str = Depends(verify_token)):
    """Joins an existing league using an invite code."""
    user = get_current_user_doc(user_id)
    if user.get("league_id"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is already in a league")
        
    league = leagues_collection.find_one({"code": join_in.code.upper()})
    
    if not league:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid League Code")
    
    # Update league members list
    leagues_collection.update_one(
        {"_id": league["_id"]},
        {"$addToSet": {"members": ObjectId(user_id)}}
    )
    
    # Update user's league_id
    league_id = str(league["_id"])
    users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"league_id": league_id}})

    return {"id": league_id, "name": league["name"], "code": league["code"]}

@app.get("/league/standings")
async def get_league_standings(user_id: str = Depends(verify_token)):
    """Retrieves current league standings and meta data."""
    user = get_current_user_doc(user_id)
    league_id = user.get("league_id")
    
    if not league_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User is not in a league")

    league = leagues_collection.find_one({"_id": ObjectId(league_id)})
    if not league:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="League not found")

    member_ids = league.get("members", [])
    members = list(users_collection.find({"_id": {"$in": member_ids}}, {"name": 1, "email": 1}))
    
    # Temporary mock data for W-L, Total Points, and Current Week Score calculation
    standings_data = []
    
    # In a real app, W-L and Total Points would be stored/calculated from historical matchups
    for member in members:
        # Mock W-L for simplicity in hackathon demo:
        wins = random.randint(0, 5)
        losses = 5 - wins
        
        # Calculate current week's score for the member (re-using habits/week endpoint logic)
        member_week_res = await get_weekly_summary(str(member["_id"]))
        current_week_score = member_week_res["score"]
        
        # Mock Total Points (sum of current week + base score)
        total_points = current_week_score["total"] + (wins * 100 + losses * 50)
        
        standings_data.append({
            "player_id": str(member["_id"]),
            "name": member["name"],
            "wins": wins,
            "losses": losses,
            "total_points": round(total_points, 1),
            "current_week_score": current_week_score
        })

    # Sort by Wins, then Total Points (Fantasy League standard)
    standings_data.sort(key=lambda x: (x["wins"], x["total_points"]), reverse=True)
    
    # Add rank
    for i, player in enumerate(standings_data):
        player["rank"] = i + 1

    return {
        "league": serialize_doc(league),
        "standings": standings_data,
        "member_count": len(member_ids)
    }

# --- League Chat Endpoints ---
@app.post("/league/chat", response_model=LeagueChatMessageOut)
async def post_league_chat(
    chat_in: LeagueChatMessageIn,
    user_id: str = Depends(verify_token)
):
    """
    Post a message to the current user's league chat.
    """
    user = get_current_user_doc(user_id)
    league_id = user.get("league_id")

    if not league_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not in a league"
        )

    now = datetime.utcnow()
    chat_doc = {
        "league_id": ObjectId(league_id),
        "user_id": ObjectId(user_id),
        "user_name": user["name"],
        "message": chat_in.message.strip(),
        "timestamp": now,
    }

    result = chats_collection.insert_one(chat_doc)
    saved = chats_collection.find_one({"_id": result.inserted_id})

    return {
        "_id": str(saved["_id"]),
        "league_id": str(saved["league_id"]),
        "user_id": str(saved["user_id"]),
        "user_name": saved["user_name"],
        "message": saved["message"],
        "timestamp": saved["timestamp"].isoformat() + "Z",
    }


@app.get("/league/chat", response_model=List[LeagueChatMessageOut])
async def get_league_chat(
    limit: int = Query(50, ge=1, le=200),
    user_id: str = Depends(verify_token)
):
    """
    Get recent league chat messages for the current user's league.
    Messages are returned in chronological order (oldest → newest).
    """
    user = get_current_user_doc(user_id)
    league_id = user.get("league_id")

    if not league_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not in a league"
        )

    cursor = (
        chats_collection
        .find({"league_id": ObjectId(league_id)})
        .sort("timestamp", -1)           
        .limit(limit)
    )

    messages = list(cursor)[::-1]        # reverse from oldest to newest

    return [
        {
            "_id": str(msg["_id"]),
            "league_id": str(msg["league_id"]),
            "user_id": str(msg["user_id"]),
            "user_name": msg["user_name"],
            "message": msg["message"],
            "timestamp": msg["timestamp"].isoformat() + "Z",
        }
        for msg in messages
    ]

# --- Matchup Endpoints (Reward/Educate) ---

@app.get("/matchup/current")
async def get_current_matchup(user_id: str = Depends(verify_token)):
    """Retrieves the current week's matchup details."""
    user = get_current_user_doc(user_id)
    league_id = user.get("league_id")
    
    if not league_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User is not in a league")

    # In a real app, this would query the matchups_collection for the current week's match.
    # For a hackathon demo, we will mock the opponent and comparison data.
    
    league = leagues_collection.find_one({"_id": ObjectId(league_id)})
    member_ids = [str(mid) for mid in league.get("members", []) if str(mid) != user_id]
    
    if not member_ids:
        return {"matchup_exists": False, "message": "Waiting for opponents to join the league."}

    # Mock opponent
    opponent_id = random.choice(member_ids)
    opponent = users_collection.find_one({"_id": ObjectId(opponent_id)})
    
    # Fetch user's current week score
    user_week_res = await get_weekly_summary(user_id)
    user_score = user_week_res["score"]
    
    # Fetch opponent's current week score
    opponent_week_res = await get_weekly_summary(opponent_id)
    opponent_score = opponent_week_res["score"]
    
    # Mocking Matchup Recap (Educate/Reward)
    recap = {
        "status": "IN_PROGRESS",
        "message": "It's a tight match! Focus on your hydration today to pull ahead.",
        "user_projection": user_score["total"] + (random.uniform(5, 15) if user_score["days_logged"] < 7 else 0),
        "opponent_projection": opponent_score["total"] + (random.uniform(5, 15) if opponent_score["days_logged"] < 7 else 0)
    }
    
    if user_score["total"] > opponent_score["total"]:
        recap["message"] = f"You're currently leading {user_score['total']:.1f} to {opponent_score['total']:.1f}! Maintain your {max(opponent_score['categories'], key=opponent_score['categories'].get, default='effort')} advantage."
    elif user_score["total"] < opponent_score["total"]:
        recap["message"] = f"You are behind. Your opponent is dominating in {max(opponent_score['categories'], key=opponent_score['categories'].get, default='all categories')}. Focus on logging your daily goals!"
    
    # Detailed category comparison
    category_comparison = {}
    all_categories = set(user_score["categories"].keys()) | set(opponent_score["categories"].keys())
    
    for cat in all_categories:
        user_cat_pts = user_score["categories"].get(cat, 0.0)
        opp_cat_pts = opponent_score["categories"].get(cat, 0.0)
        category_comparison[cat] = {
            "user": round(user_cat_pts, 1),
            "opponent": round(opp_cat_pts, 1),
            "leading": "user" if user_cat_pts > opp_cat_pts else ("opponent" if opp_cat_pts > user_cat_pts else "tie")
        }

    return {
        "matchup_exists": True,
        "week": datetime.now().isocalendar()[1],
        "user": {"name": user["name"], "score": user_score},
        "opponent": {"name": opponent["name"], "score": opponent_score},
        "recap": recap,
        "comparison": category_comparison
    }

# --- Insights Endpoints (Educate/Track) ---

@app.get("/insights/summary")
async def get_insights_summary(user_id: str = Depends(verify_token)):
    """Retrieves key metrics and actionable recommendations."""
    user = get_current_user_doc(user_id)
    
    # 1. Streak (Track)
    streak = calculate_streak(user_id)
    
    # 2. Average Score
    last_7_days = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_entries = list(habit_entries_collection.find({
        "user_id": ObjectId(user_id),
        "date": {"$gte": last_7_days}
    }))
    
    avg_daily_score = sum(e["total_points"] for e in recent_entries) / len(recent_entries) if recent_entries else 0.0
    
    # 3. Recommendations (Educate)
    recommendations = []
    if avg_daily_score < 10:
        recommendations.append({
            "priority": "high",
            "title": "Boost Daily Consistency",
            "description": "Your average daily score is low. Focus on consistently logging at least one habit daily to build momentum.",
            "category": "general"
        })
    
    # Example: Check for low sleep average
    total_sleep = sum(e["entry"].get("sleep", 0) for e in recent_entries)
    avg_sleep = total_sleep / len(recent_entries) if recent_entries and total_sleep > 0 else 0
    if avg_sleep > 0 and avg_sleep < 6.5 and user.get("goals", {}).get("sleep", 8.0) > 0:
        recommendations.append({
            "priority": "medium",
            "title": "Prioritize Sleep Hygiene",
            "description": f"Your average sleep over the last week is {avg_sleep:.1f} hours, well below your goal. Try reducing screen time 30 mins before bed.",
            "category": "sleep"
        })
    
    # Example: Check for exercise streak
    exercise_days = sum(1 for e in recent_entries if e["entry"].get("exercise", 0) >= 0.5)
    if exercise_days < 3:
         recommendations.append({
            "priority": "low",
            "title": "Increase Activity Frequency",
            "description": f"You've logged exercise only {exercise_days} times this week. Aim for at least 3-4 sessions for optimal heart health.",
            "category": "exercise"
        })

    return {
        "streak": streak,
        "avg_daily_score": round(avg_daily_score, 1),
        "recommendations": recommendations,
        "goals": user.get("goals", {})
    }

@app.get("/insights/coach-tip")
async def get_daily_tip(user_id: str = Depends(verify_token)):
    """Get a daily coach tip, using the logic defined earlier."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    entry = habit_entries_collection.find_one({
        "user_id": ObjectId(user_id),
        "date": today
    })
    
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    goals = user.get("goals", {})
    
    # The get_coach_tip logic handles the "Educate" component by being context-aware
    if entry:
        return get_coach_tip(entry["entry"], goals)
    else:
        return {"category": "general", "tip": random.choice(COACH_TIPS["general"])}

# --- Goals Endpoint ---

class UpdateGoalsIn(BaseModel):
    goals: HabitGoal

@app.put("/user/goals")
async def update_user_goals(goals_in: UpdateGoalsIn, user_id: str = Depends(verify_token)):
    """Allows a user to update their habit goals."""
    
    update_result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"goals": goals_in.goals.dict()}}
    )
    
    if update_result.modified_count == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Goals not updated or no change detected")

    return {"message": "Goals updated successfully"}