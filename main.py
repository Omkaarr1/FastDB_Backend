import os
import io
import json
import traceback
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import uuid

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import ORJSONResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from passlib.context import CryptContext
from jose import JWTError, jwt
import lmdb

# --------------------------------------------------
# 1. Environment Variables & IST Timestamp
# --------------------------------------------------
DATABASE_PATH = os.getenv("DATABASE_PATH", "./lmdb_data")
SECRET_KEY = os.getenv("SECRET_KEY", "my-secret-key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 60
IST = timezone(timedelta(hours=5, minutes=30))

# --------------------------------------------------
# 2. Global LMDB Environment & Sub–databases
# --------------------------------------------------
LMDB_ENV = None
DB_USERS = None
DB_REQUESTS = None
DB_APPROVER_ACTIONS = None
DB_ERROR_LOGS = None
DB_TOKENS = None  # Sub-database for active tokens

def init_lmdb():
    """Initialize LMDB environment and open sub-databases."""
    global LMDB_ENV, DB_USERS, DB_REQUESTS, DB_APPROVER_ACTIONS, DB_ERROR_LOGS, DB_TOKENS
    if not os.path.exists(DATABASE_PATH):
        os.makedirs(DATABASE_PATH)
    LMDB_ENV = lmdb.open(DATABASE_PATH, map_size=1024 * 1024 * 1024, max_dbs=10)
    DB_USERS = LMDB_ENV.open_db(b'users')
    DB_REQUESTS = LMDB_ENV.open_db(b'requests')
    DB_APPROVER_ACTIONS = LMDB_ENV.open_db(b'approver_actions')
    DB_ERROR_LOGS = LMDB_ENV.open_db(b'error_logs')
    DB_TOKENS = LMDB_ENV.open_db(b'active_tokens')

def get_next_id(txn, db_handle):
    """Generate an auto–incremented ID for a given LMDB sub–database."""
    counter_key = b'__counter__'
    current = txn.get(counter_key, db=db_handle)
    if current is None:
        next_id = 1
    else:
        next_id = int(current.decode()) + 1
    txn.put(counter_key, str(next_id).encode(), db=db_handle)
    return next_id

# --------------------------------------------------
# 3. Database Helper Functions
# --------------------------------------------------
# Users
def insert_user(user: dict) -> dict:
    with LMDB_ENV.begin(write=True, db=DB_USERS) as txn:
        user_id = get_next_id(txn, DB_USERS)
        user['id'] = user_id
        txn.put(str(user_id).encode(), json.dumps(user).encode(), db=DB_USERS)
    return user

def get_user_by_id(user_id: int) -> Optional[dict]:
    with LMDB_ENV.begin(db=DB_USERS) as txn:
        data = txn.get(str(user_id).encode(), db=DB_USERS)
        if data:
            return json.loads(data.decode())
    return None

def get_user_by_username(username: str) -> Optional[dict]:
    with LMDB_ENV.begin(db=DB_USERS) as txn:
        with txn.cursor(db=DB_USERS) as cursor:
            for key, value in cursor:
                if key == b'__counter__':
                    continue
                user = json.loads(value.decode())
                if user.get("username") == username:
                    return user
    return None

def list_all_users() -> List[dict]:
    users = []
    with LMDB_ENV.begin(db=DB_USERS) as txn:
        with txn.cursor(db=DB_USERS) as cursor:
            for key, value in cursor:
                if key == b'__counter__':
                    continue
                users.append(json.loads(value.decode()))
    return users

def update_user(user: dict) -> None:
    with LMDB_ENV.begin(write=True, db=DB_USERS) as txn:
        txn.put(str(user['id']).encode(), json.dumps(user).encode(), db=DB_USERS)

# Requests
def insert_request(req: dict) -> dict:
    with LMDB_ENV.begin(write=True, db=DB_REQUESTS) as txn:
        req_id = get_next_id(txn, DB_REQUESTS)
        req['id'] = req_id
        txn.put(str(req_id).encode(), json.dumps(req).encode(), db=DB_REQUESTS)
    return req

def get_request_by_id(request_id: int) -> Optional[dict]:
    with LMDB_ENV.begin(db=DB_REQUESTS) as txn:
        data = txn.get(str(request_id).encode(), db=DB_REQUESTS)
        if data:
            return json.loads(data.decode())
    return None

def update_request(req: dict) -> None:
    with LMDB_ENV.begin(write=True, db=DB_REQUESTS) as txn:
        txn.put(str(req['id']).encode(), json.dumps(req).encode(), db=DB_REQUESTS)

def list_all_requests() -> List[dict]:
    reqs = []
    with LMDB_ENV.begin(db=DB_REQUESTS) as txn:
        with txn.cursor(db=DB_REQUESTS) as cursor:
            for key, value in cursor:
                if key == b'__counter__':
                    continue
                reqs.append(json.loads(value.decode()))
    return reqs

# Approver Actions
def insert_approver_action(action: dict) -> dict:
    with LMDB_ENV.begin(write=True, db=DB_APPROVER_ACTIONS) as txn:
        action_id = get_next_id(txn, DB_APPROVER_ACTIONS)
        action['id'] = action_id
        txn.put(str(action_id).encode(), json.dumps(action).encode(), db=DB_APPROVER_ACTIONS)
    return action

def get_approver_action(request_id: int, approver_id: int) -> Optional[dict]:
    with LMDB_ENV.begin(db=DB_APPROVER_ACTIONS) as txn:
        with txn.cursor(db=DB_APPROVER_ACTIONS) as cursor:
            for key, value in cursor:
                if key == b'__counter__':
                    continue
                action = json.loads(value.decode())
                if action.get("request_id") == request_id and action.get("approver_id") == approver_id:
                    return action
    return None

def update_approver_action(action: dict) -> None:
    with LMDB_ENV.begin(write=True, db=DB_APPROVER_ACTIONS) as txn:
        txn.put(str(action['id']).encode(), json.dumps(action).encode(), db=DB_APPROVER_ACTIONS)

def list_approver_actions_by_request(request_id: int) -> List[dict]:
    actions = []
    with LMDB_ENV.begin(db=DB_APPROVER_ACTIONS) as txn:
        with txn.cursor(db=DB_APPROVER_ACTIONS) as cursor:
            for key, value in cursor:
                if key == b'__counter__':
                    continue
                action = json.loads(value.decode())
                if action.get("request_id") == request_id:
                    actions.append(action)
    return actions

def delete_approver_actions_by_request(request_id: int) -> None:
    with LMDB_ENV.begin(write=True, db=DB_APPROVER_ACTIONS) as txn:
        with txn.cursor(db=DB_APPROVER_ACTIONS) as cursor:
            keys_to_delete = []
            for key, value in cursor:
                if key == b'__counter__':
                    continue
                action = json.loads(value.decode())
                if action.get("request_id") == request_id:
                    keys_to_delete.append(key)
            for k in keys_to_delete:
                txn.delete(k, db=DB_APPROVER_ACTIONS)

# Error Logs
def insert_error_log(log: dict) -> None:
    with LMDB_ENV.begin(write=True, db=DB_ERROR_LOGS) as txn:
        log_id = get_next_id(txn, DB_ERROR_LOGS)
        log['id'] = log_id
        txn.put(str(log_id).encode(), json.dumps(log).encode(), db=DB_ERROR_LOGS)

# --------------------------------------------------
# 4. Password Hashing & JWT Setup
# --------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    # Embed a unique session identifier (jti)
    to_encode.update({
        "exp": expire,
        "jti": str(uuid.uuid4())
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --------------------------------------------------
# 5. Persistent Token Store Helper Functions (Updated for multi-login IP/user-agent)
# --------------------------------------------------
def store_token(token: str, details: dict):
    """
    Stores token details using the token's 'jti' (session ID) as the key.
    'details' can include user_id, ip_address, user_agent, login time, etc.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        session_id = payload.get("jti")
        if not session_id:
            raise ValueError("Token missing session identifier (jti)")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    with LMDB_ENV.begin(write=True, db=DB_TOKENS) as txn:
        txn.put(session_id.encode(), json.dumps(details).encode(), db=DB_TOKENS)

def remove_token(token: str):
    """
    Removes a token from the DB_TOKENS sub-database using its session ID (jti).
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        session_id = payload.get("jti")
    except JWTError:
        return  # Nothing to remove if token is invalid.
    if session_id:
        with LMDB_ENV.begin(write=True, db=DB_TOKENS) as txn:
            txn.delete(session_id.encode(), db=DB_TOKENS)

def get_token_details(token: str) -> Optional[dict]:
    """
    Retrieves token details from the DB_TOKENS sub-database by its session ID (jti).
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        session_id = payload.get("jti")
    except JWTError:
        return None
    if not session_id:
        return None
    with LMDB_ENV.begin(db=DB_TOKENS) as txn:
        data = txn.get(session_id.encode(), db=DB_TOKENS)
        if data:
            return json.loads(data.decode())
    return None

def list_tokens_by_user(user_id: int) -> List[dict]:
    sessions = []
    with LMDB_ENV.begin(db=DB_TOKENS) as txn:
        with txn.cursor(db=DB_TOKENS) as cursor:
            for key, value in cursor:
                # Skip the counter key
                if key == b'__counter__':
                    continue

                details = json.loads(value.decode())
                if details.get("user_id") == user_id:
                    sessions.append({
                        "session_id": key.decode(),
                        "login_time": details.get("created_at"),
                        "ip_address": details.get("ip_address"),
                        "user_agent": details.get("user_agent")
                    })
    return sessions

def remove_tokens_by_user(user_id: int):
    """
    Removes all active sessions for a given user.
    """
    with LMDB_ENV.begin(write=True, db=DB_TOKENS) as txn:
        with txn.cursor(db=DB_TOKENS) as cursor:
            keys_to_delete = []
            for key, value in cursor:
                # Skip the LMDB counter key
                if key == b'__counter__':
                    continue
                details = json.loads(value.decode())
                if details.get("user_id") == user_id:
                    keys_to_delete.append(key)
            for k in keys_to_delete:
                txn.delete(k, db=DB_TOKENS)

# --------------------------------------------------
# 6. Pydantic Schemas
# --------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

class UserBase(BaseModel):
    name: str
    role: List[int] = [0]

class UserCreate(UserBase):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    username: str
    class Config:
        orm_mode = True

class ApproverActionResponse(BaseModel):
    approver_id: int
    approved: Optional[str] = None
    action_time: Optional[str] = None
    received_at: Optional[str] = None
    comment: Optional[str] = None
    class Config:
        orm_mode = True

class RequestResponse(BaseModel):
    id: int
    initiator_id: int
    supervisor_id: int
    subject: str
    description: str
    area: str
    project: str
    tower: str
    department: str
    references: Optional[str] = None
    priority: str
    approvers: List[int]
    current_approver_index: int
    status: str
    created_at: str
    updated_at: str
    last_action: Optional[str] = None
    supervisor_approved_at: Optional[str] = None
    initiator_name: Optional[str] = None
    supervisor_name: Optional[str] = None
    pending_at: Optional[str] = None
    approver_actions: Optional[List[ApproverActionResponse]] = None
    approval_hierarchy: Optional[List[dict]] = None
    files: Optional[List[dict]] = []
    file_url: Optional[str] = None
    file_display_name: Optional[str] = None

class ApprovalAction(BaseModel):
    request_id: int
    approved: bool
    comment: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    login_time: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

# --------------------------------------------------
# 7. FastAPI App, CORS & Static Files
# --------------------------------------------------
app = FastAPI(title="Request Management System", default_response_class=ORJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "nfa_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.mount("/files", StaticFiles(directory=UPLOAD_FOLDER), name="files")

# --------------------------------------------------
# 8. Global Exception Handler to Log Errors
# --------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log = {
        "endpoint": str(request.url),
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "created_at": datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    }
    try:
        insert_error_log(log)
    except Exception:
        pass
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# --------------------------------------------------
# 9. Dependencies & get_current_user
# --------------------------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    # Check if the token exists in the persistent token store.
    details = get_token_details(token)
    if not details:
        raise HTTPException(status_code=401, detail="Token is not active. Please login.")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=int(user_id))
    except JWTError:
        raise credentials_exception
    user = get_user_by_id(token_data.user_id)
    if not user:
        raise credentials_exception
    return user

# --------------------------------------------------
# 10. Helper Functions to Format Request Responses
# --------------------------------------------------
async def can_view_request(r: dict, user: dict) -> bool:
    if user["id"] == r["initiator_id"] or user["id"] == r["supervisor_id"]:
        return True
    if r["status"] in ("APPROVED", "REJECTED") and user["id"] in r["approvers"]:
        return True
    if r["status"] == "NEW":
        return False
    if r["status"] == "IN_PROGRESS":
        if r["current_approver_index"] < len(r["approvers"]) and user["id"] == r["approvers"][r["current_approver_index"]]:
            return True
    return False

async def to_request_response(r: dict) -> RequestResponse:
    initiator = get_user_by_id(r["initiator_id"])
    supervisor = get_user_by_id(r["supervisor_id"])
    initiator_name = initiator["name"] if initiator else "Unknown"
    supervisor_name = supervisor["name"] if supervisor else "Unknown"
    pending_at = None
    approval_hierarchy = []

    # Supervisor row
    approval_hierarchy.append({
        "role": "Supervisor",
        "user_id": r["supervisor_id"],
        "name": supervisor_name,
        "approved": "Approved" if r.get("supervisor_approved") is True else ("Declined" if r.get("supervisor_approved") is False else "Pending"),
        "received_at": r["created_at"],
        "action_time": r.get("supervisor_approved_at"),
        "comment": r.get("supervisor_comment")
    })

    if r["status"] == "IN_PROGRESS":
        for approver_id in r["approvers"]:
            action_obj = get_approver_action(r["id"], approver_id)
            user_obj = get_user_by_id(approver_id)
            if action_obj:
                approval_hierarchy.append({
                    "role": "Approver",
                    "user_id": approver_id,
                    "name": user_obj["name"] if user_obj else "Unknown",
                    "approved": action_obj.get("approved"),
                    "received_at": action_obj.get("received_at") if action_obj.get("received_at") else "N/A",
                    "action_time": action_obj.get("action_time") if action_obj.get("action_time") else "Pending",
                    "comment": action_obj.get("comment")
                })
            else:
                approval_hierarchy.append({
                    "role": "Approver",
                    "user_id": approver_id,
                    "name": user_obj["name"] if user_obj else "Unknown",
                    "approved": "Pending",
                    "received_at": "N/A",
                    "action_time": "Pending",
                    "comment": None
                })
                break
    elif r["status"] == "REJECTED":
        for approver_id in r["approvers"]:
            action_obj = get_approver_action(r["id"], approver_id)
            if action_obj:
                user_obj = get_user_by_id(approver_id)
                approval_hierarchy.append({
                    "role": "Approver",
                    "user_id": approver_id,
                    "name": user_obj["name"] if user_obj else "Unknown",
                    "approved": action_obj.get("approved"),
                    "received_at": action_obj.get("received_at") if action_obj.get("received_at") else "N/A",
                    "action_time": action_obj.get("action_time") if action_obj.get("action_time") else "N/A",
                    "comment": action_obj.get("comment")
                })
            else:
                break
    elif r["status"] == "APPROVED":
        for approver_id in r["approvers"]:
            action_obj = get_approver_action(r["id"], approver_id)
            user_obj = get_user_by_id(approver_id)
            approval_hierarchy.append({
                "role": "Approver",
                "user_id": approver_id,
                "name": user_obj["name"] if user_obj else "Unknown",
                "approved": action_obj.get("approved") if action_obj else "Pending",
                "received_at": action_obj.get("received_at") if action_obj and action_obj.get("received_at") else "N/A",
                "action_time": action_obj.get("action_time") if action_obj and action_obj.get("action_time") else "Pending",
                "comment": action_obj.get("comment") if action_obj else None
            })

    if r["status"] == "IN_PROGRESS":
        if r["current_approver_index"] < len(r["approvers"]):
            next_approver = get_user_by_id(r["approvers"][r["current_approver_index"]])
            if next_approver:
                pending_at = f"Approver: {next_approver['name']} ({next_approver['role']})"
            else:
                pending_at = "Approver: Unknown"
    elif r["status"] == "NEW":
        pending_at = "Supervisor"
    else:
        pending_at = None

    actions = []
    for act in list_approver_actions_by_request(r["id"]):
        actions.append({
            "approver_id": act.get("approver_id"),
            "approved": act.get("approved"),
            "action_time": act.get("action_time"),
            "received_at": act.get("received_at"),
            "comment": act.get("comment")
        })

    response = {
        "id": r["id"],
        "initiator_id": r["initiator_id"],
        "supervisor_id": r["supervisor_id"],
        "subject": r["subject"],
        "description": r["description"],
        "area": r["area"],
        "project": r["project"],
        "tower": r["tower"],
        "department": r["department"],
        "references": r.get("references"),
        "priority": r["priority"],
        "approvers": r["approvers"],
        "current_approver_index": r["current_approver_index"],
        "status": r["status"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "last_action": r["last_action"],
        "supervisor_approved_at": r.get("supervisor_approved_at"),
        "initiator_name": initiator_name,
        "supervisor_name": supervisor_name,
        "pending_at": pending_at,
        "approver_actions": actions,
        "approval_hierarchy": approval_hierarchy,
        "files": r.get("files", [])
    }
    return RequestResponse(**response)

def displayOrNA(value):
    if value is None or str(value).strip() == "":
        return "N/A"
    return str(value)

# --------------------------------------------------
# 11. Auth & User Endpoints
# --------------------------------------------------
@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    if get_user_by_username(user_data.username):
        raise HTTPException(status_code=400, detail="User with this username already exists.")
    new_user = {
        "username": user_data.username,
        "name": user_data.name,
        "role": user_data.role,
        "hashed_password": get_password_hash(user_data.password)
    }
    new_user = insert_user(new_user)
    return new_user

@app.post("/login", response_model=Token)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": str(user["id"])})
    # Gather IP and User-Agent
    client_ip = request.client.host if request.client else "Unknown"
    user_agent = request.headers.get("User-Agent", "Unknown")
    # Store the token (using its session id) along with login details (including IP and user-agent).
    store_token(access_token, {
        "user_id": user["id"],
        "created_at": datetime.utcnow().isoformat(),
        "ip_address": client_ip,
        "user_agent": user_agent
    })
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    remove_token(token)
    return {"detail": "Successfully logged out."}

@app.post("/logout_all")
async def logout_all(current_user: dict = Depends(get_current_user)):
    remove_tokens_by_user(current_user["id"])
    return {"detail": "Logged out from all sessions."}

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(current_user: dict = Depends(get_current_user)):
    sessions = list_tokens_by_user(current_user["id"])
    # Convert each record into SessionInfo
    result = []
    for s in sessions:
        result.append(SessionInfo(
            session_id=s.get("session_id"),
            login_time=s.get("login_time"),
            ip_address=s.get("ip_address"),
            user_agent=s.get("user_agent")
        ))
    return result

@app.get("/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/users/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, current_user: dict = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/", response_model=List[UserResponse])
async def list_users(current_user: dict = Depends(get_current_user)):
    return list_all_users()

# --------------------------------------------------
# 12. Request Endpoints
# --------------------------------------------------
@app.post("/requests/", response_model=RequestResponse, status_code=status.HTTP_201_CREATED)
async def create_request(
    initiator_id: int = Form(...),
    supervisor_id: int = Form(...),
    subject: str = Form(...),
    description: str = Form(...),
    area: str = Form(...),
    project: str = Form(...),
    tower: str = Form(...),
    department: str = Form(...),
    references: str = Form(...),
    priority: str = Form(...),
    approvers: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    current_user: dict = Depends(get_current_user)
):
    if current_user["id"] != initiator_id:
        raise HTTPException(status_code=403, detail="You can only create NFAs for yourself.")
    if supervisor_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot choose yourself as supervisor.")
    supervisor = get_user_by_id(supervisor_id)
    if not supervisor:
        raise HTTPException(status_code=404, detail="Supervisor not found.")

    try:
        approvers_list = json.loads(approvers)
        # Ensure supervisor_id is not in the approver list
        approvers_list = [x for x in approvers_list if x != supervisor_id]
        if not isinstance(approvers_list, list) or not all(isinstance(x, int) for x in approvers_list):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Approvers must be a JSON list of user IDs.")

    current_time_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    new_req = {
        "initiator_id": initiator_id,
        "supervisor_id": supervisor_id,
        "subject": subject,
        "description": description,
        "area": area,
        "project": project,
        "tower": tower,
        "department": department,
        "references": references,
        "priority": priority,
        "approvers": approvers_list,
        "current_approver_index": 0,
        "status": "NEW",
        "created_at": current_time_str,
        "updated_at": current_time_str,
        "last_action": "NFA initiated.",
        "supervisor_approved_at": None,
        "supervisor_approved": None,
        "supervisor_comment": None,
        "files": []
    }
    new_req = insert_request(new_req)

    file_records = []
    if files is not None:
        for file in files:
            original_filename = file.filename or "unnamed_file"
            sanitized_filename = original_filename.replace(" ", "_")
            timestamp = datetime.now(IST).strftime('%Y%m%d%H%M%S')
            new_filename = f"{new_req['id']}_{timestamp}_{sanitized_filename}"
            file_location = os.path.join(UPLOAD_FOLDER, new_filename)
            try:
                file.file.seek(0)
                content = await file.read()
                with open(file_location, "wb") as f:
                    f.write(content)
                file_record = {"file_url": f"/files/{new_filename}", "file_display_name": original_filename}
                file_records.append(file_record)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    new_req["files"] = file_records
    update_request(new_req)
    return await to_request_response(new_req)

@app.get("/requests/", response_model=List[RequestResponse])
async def list_requests(
    note_id: Optional[int] = None,
    date: Optional[str] = None,
    initiator: Optional[str] = None,
    filter: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    all_requests = list_all_requests()
    if note_id:
        all_requests = [r for r in all_requests if r["id"] == note_id]

    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            next_day = date_obj + timedelta(days=1)
            def within_date(r):
                r_date = datetime.strptime(r["created_at"], "%d-%m-%Y %H:%M")
                return date_obj <= r_date < next_day
            all_requests = [r for r in all_requests if within_date(r)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    if initiator:
        def match_initiator(r):
            user_ = get_user_by_id(r["initiator_id"])
            return user_ and initiator.lower() in user_["name"].lower()
        all_requests = [r for r in all_requests if match_initiator(r)]

    if filter:
        f = filter.upper()
        if f == "PENDING":
            all_requests = [r for r in all_requests if r["status"] in ("NEW", "IN_PROGRESS")]
        elif f == "APPROVED":
            all_requests = [r for r in all_requests if r["status"] == "APPROVED"]

    visible = []
    for r in all_requests:
        if await can_view_request(r, current_user):
            visible.append(r)
    responses = []
    for r in visible:
        responses.append(await to_request_response(r))
    return responses

@app.get("/requests/{request_id}/pdf")
async def download_pdf(request_id: int, current_user: dict = Depends(get_current_user)):
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="NFA not found")
    if req["status"] != "APPROVED":
        raise HTTPException(status_code=400, detail="PDF can only be downloaded for approved NFAs.")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40

    c.setFont("Helvetica-Bold", 18)
    details_title = "NFA Request Details"
    title_width = c.stringWidth(details_title, "Helvetica-Bold", 18)
    c.drawString((width - title_width) / 2, height - margin, details_title)
    y_position = height - margin - 30

    details = [
        ("Initiator ID", req.get("initiator_id")),
        ("Supervisor ID", req.get("supervisor_id")),
        ("Subject", req.get("subject")),
        ("Description", req.get("description")),
        ("Area", req.get("area")),
        ("Project", req.get("project")),
        ("Tower", req.get("tower")),
        ("Department", req.get("department")),
        ("Priority", req.get("priority")),
        ("References", req.get("references"))
    ]
    for label, value in details:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_position, f"{label}:")
        c.setFont("Helvetica", 12)
        c.drawString(margin + 150, y_position, displayOrNA(value))
        y_position -= 20

    y_position -= 30
    c.setFont("Helvetica-Bold", 18)
    table_title = "NFA Approval Summary"
    table_title_width = c.stringWidth(table_title, "Helvetica-Bold", 18)
    c.drawString((width - table_title_width) / 2, y_position, table_title)
    y_position -= 30

    row_height = 20
    col_widths = [30, 150, 120, 100, 100]
    x_positions = [margin]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    c.setFillColorRGB(0.85, 0.85, 0.85)
    c.rect(margin, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 10)
    headers = ["S. No.", "Particular", "Name & Designation", "Received Date", "Approved Date"]
    for i, header in enumerate(headers):
        c.drawString(x_positions[i] + 2, y_position - row_height + 5, header)
    temp_x = margin
    for w in col_widths:
        c.rect(temp_x, y_position - row_height, w, row_height, stroke=1, fill=0)
        temp_x += w

    rows = []
    initiator = get_user_by_id(req["initiator_id"])
    initiator_name = initiator["name"] if initiator else "Unknown"
    rows.append({
        "sno": "1",
        "particular": "Initiator",
        "name": initiator_name,
        "received": req["created_at"],
        "approved": "-"
    })
    supervisor = get_user_by_id(req["supervisor_id"])
    supervisor_name = supervisor["name"] if supervisor else "Unknown"
    sup_approved = req.get("supervisor_approved_at") if req.get("supervisor_approved_at") else "Pending"
    rows.append({
        "sno": "2",
        "particular": "Supervisor",
        "name": supervisor_name,
        "received": req["created_at"],
        "approved": sup_approved
    })
    sno = 3
    for approver_id in req["approvers"]:
        user_obj = get_user_by_id(approver_id)
        name = user_obj["name"] if user_obj else "Unknown"
        action_obj = get_approver_action(req["id"], approver_id)
        received_date = action_obj.get("received_at") if (action_obj and action_obj.get("received_at")) else "N/A"
        approved_date = action_obj.get("action_time") if (action_obj and action_obj.get("action_time")) else "Pending"
        rows.append({
            "sno": str(sno),
            "particular": "Approver",
            "name": name,
            "received": received_date,
            "approved": approved_date
        })
        sno += 1

    c.setFont("Helvetica", 10)
    current_y = y_position - row_height
    for row in rows:
        current_y -= row_height
        current_x = margin
        for w in col_widths:
            c.rect(current_x, current_y, w, row_height, stroke=1, fill=0)
            current_x += w
        c.drawString(margin + 2, current_y + 5, row["sno"])
        c.drawString(x_positions[1] + 2, current_y + 5, row["particular"])
        c.drawString(x_positions[2] + 2, current_y + 5, row["name"])
        c.drawString(x_positions[3] + 2, current_y + 5, row["received"])
        c.drawString(x_positions[4] + 2, current_y + 5, row["approved"])

    c.setFillColorRGB(0, 0, 1)
    c.setFont("Helvetica-Bold", 12)
    note = "This is a system generated pdf, no need of signatures"
    note_width = c.stringWidth(note, "Helvetica-Bold", 12)
    c.drawString((width - note_width) / 2, 20, note)
    c.setFillColorRGB(0, 0, 0)

    c.showPage()
    c.save()
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="nfa_{request_id}.pdf"'}
    )

@app.post("/upload-file/{request_id}")
async def upload_files_for_request(
    request_id: int,
    files: Optional[List[UploadFile]] = File(None),
    current_user: dict = Depends(get_current_user)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req["initiator_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to upload files for this request")

    file_records = []
    for file in files:
        original_filename = file.filename or "unnamed_file"
        sanitized_filename = original_filename.replace(" ", "_")
        ext = sanitized_filename.split('.')[-1].lower() if '.' in sanitized_filename else ''
        if ext == "pdf":
            subfolder = "pdf"
        elif ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
            subfolder = "image"
        else:
            subfolder = "others"
        subfolder_path = os.path.join(UPLOAD_FOLDER, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        timestamp = datetime.now(IST).strftime('%Y%m%d%H%M%S')
        new_filename = f"{request_id}_{timestamp}_{sanitized_filename}"
        file_location = os.path.join(subfolder_path, new_filename)
        try:
            file.file.seek(0)
            content = await file.read()
            with open(file_location, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        file_record = {"file_url": f"/files/{subfolder}/{new_filename}", "file_display_name": original_filename}
        file_records.append(file_record)

    if "files" not in req or not isinstance(req["files"], list):
        req["files"] = []
    req["files"].extend(file_records)
    update_request(req)
    return {"files": req["files"]}

@app.post("/requests/supervisor-review", response_model=RequestResponse)
async def supervisor_review(action: ApprovalAction, current_user: dict = Depends(get_current_user)):
    req = get_request_by_id(action.request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req["supervisor_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="You are not authorized to perform supervisor review for this request")
    if req.get("supervisor_approved") is not None:
        raise HTTPException(status_code=400, detail="Supervisor has already reviewed this request")

    current_time_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    req["supervisor_approved"] = action.approved
    req["supervisor_approved_at"] = current_time_str
    req["supervisor_comment"] = action.comment
    req["updated_at"] = current_time_str

    if action.approved:
        req["status"] = "IN_PROGRESS"
        req["last_action"] = f"Supervisor approved at {current_time_str}"
    else:
        req["status"] = "REJECTED"
        req["last_action"] = f"Supervisor rejected at {current_time_str}"

    update_request(req)
    return await to_request_response(req)

@app.post("/requests/approve", response_model=RequestResponse)
async def approver_action(action: ApprovalAction, current_user: dict = Depends(get_current_user)):
    req = get_request_by_id(action.request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req["status"] != "IN_PROGRESS":
        raise HTTPException(status_code=400, detail="Request is not in progress for approval")
    if req["current_approver_index"] >= len(req["approvers"]):
        raise HTTPException(status_code=400, detail="No pending approver action for this request")

    current_approver_id = req["approvers"][req["current_approver_index"]]
    if current_user["id"] != current_approver_id:
        raise HTTPException(status_code=403, detail="You are not authorized to approve this request")

    existing_action = get_approver_action(req["id"], current_user["id"])
    if existing_action is not None:
        raise HTTPException(status_code=400, detail="Approver has already taken action on this request")

    current_time_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    new_action = {
        "request_id": req["id"],
        "approver_id": current_user["id"],
        "approved": "Approved" if action.approved else "Declined",
        "received_at": current_time_str,
        "action_time": current_time_str,
        "comment": action.comment
    }
    insert_approver_action(new_action)

    if action.approved:
        req["current_approver_index"] += 1
        req["last_action"] = f"Approver {current_user['id']} approved at {current_time_str}"
        if req["current_approver_index"] >= len(req["approvers"]):
            req["status"] = "APPROVED"
    else:
        req["status"] = "REJECTED"
        req["last_action"] = f"Approver {current_user['id']} declined at {current_time_str}"

    req["updated_at"] = current_time_str
    update_request(req)
    return await to_request_response(req)

@app.post("/requests/{request_id}/reinitiate", response_model=RequestResponse)
async def reinitiate_request(
    request_id: int,
    edit_details: bool = Form(False),
    subject: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    area: Optional[str] = Form(None),
    project: Optional[str] = Form(None),
    tower: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    references: Optional[str] = Form(None),
    priority: Optional[str] = Form(None),
    approvers: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    current_user: dict = Depends(get_current_user)
):
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req["initiator_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="You are not allowed to reinitiate this request")
    if req["status"] != "REJECTED":
        raise HTTPException(status_code=400, detail="Only declined (REJECTED) requests can be re-initiated.")

    current_time_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    if edit_details:
        if not (subject and description and area and project and tower and department and references and priority and approvers):
            raise HTTPException(status_code=400, detail="All fields are required for editing details.")
        try:
            approvers_list = json.loads(approvers)
            approvers_list = [x for x in approvers_list if x != req["supervisor_id"]]
            if not isinstance(approvers_list, list) or not all(isinstance(x, int) for x in approvers_list):
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="Approvers must be a JSON list of user IDs.")

        req["subject"] = subject
        req["description"] = description
        req["area"] = area
        req["project"] = project
        req["tower"] = tower
        req["department"] = department
        req["references"] = references
        req["priority"] = priority
        req["approvers"] = approvers_list

    req["status"] = "NEW"
    req["current_approver_index"] = 0
    req["supervisor_approved"] = None
    req["supervisor_approved_at"] = None
    req["supervisor_comment"] = None
    req["last_action"] = f"Request re-initiated at {current_time_str}"
    delete_approver_actions_by_request(req["id"])

    if files is not None:
        if "files" not in req or not isinstance(req["files"], list):
            req["files"] = []
        for file in files:
            try:
                file.file.seek(0)
                content = await file.read()
                timestamp = datetime.now(IST).strftime('%Y%m%d%H%M%S')
                new_filename = f"{req['id']}_{timestamp}"
                ext = file.filename.split('.')[-1] if file.filename and '.' in file.filename else ''
                if ext:
                    new_filename += f".{ext}"
                file_location = os.path.join(UPLOAD_FOLDER, new_filename)
                with open(file_location, "wb") as f:
                    f.write(content)
                file_record = {"file_url": f"/files/{new_filename}", "file_display_name": file.filename}
                req["files"].append(file_record)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    req["updated_at"] = current_time_str
    update_request(req)
    return await to_request_response(req)

# --------------------------------------------------
# 13. Admin Endpoints
# --------------------------------------------------
@app.post("/cleardb")
async def clear_db(current_user: dict = Depends(get_current_user)):
    if 2 not in current_user["role"]:
        raise HTTPException(status_code=403, detail="Only admin users can clear the database.")
    with LMDB_ENV.begin(write=True, db=DB_APPROVER_ACTIONS) as txn:
        with txn.cursor(db=DB_APPROVER_ACTIONS) as cursor:
            for key, _ in cursor:
                if key != b'__counter__':
                    txn.delete(key, db=DB_APPROVER_ACTIONS)
            txn.put(b'__counter__', b'0', db=DB_APPROVER_ACTIONS)

    with LMDB_ENV.begin(write=True, db=DB_REQUESTS) as txn:
        with txn.cursor(db=DB_REQUESTS) as cursor:
            for key, _ in cursor:
                if key != b'__counter__':
                    txn.delete(key, db=DB_REQUESTS)
            txn.put(b'__counter__', b'0', db=DB_REQUESTS)

    with LMDB_ENV.begin(write=True, db=DB_USERS) as txn:
        with txn.cursor(db=DB_USERS) as cursor:
            for key, _ in cursor:
                if key != b'__counter__':
                    txn.delete(key, db=DB_USERS)
            txn.put(b'__counter__', b'0', db=DB_USERS)

    # Also clear the active_tokens sub-db
    with LMDB_ENV.begin(write=True, db=DB_TOKENS) as txn:
        with txn.cursor(db=DB_TOKENS) as cursor:
            for key, _ in cursor:
                if key != b'__counter__':
                    txn.delete(key, db=DB_TOKENS)
            txn.put(b'__counter__', b'0', db=DB_TOKENS)

    return {"detail": "Database cleared successfully."}

@app.get("/transferdb", response_class=FileResponse)
async def transfer_db(current_user: dict = Depends(get_current_user)):
    if 2 not in current_user["role"]:
        raise HTTPException(status_code=403, detail="Only admin users can transfer the database.")
    db_file = os.path.join(DATABASE_PATH, "data.mdb")
    if not os.path.exists(db_file):
        raise HTTPException(status_code=404, detail="Database file not found.")
    return FileResponse(path=db_file, filename="data.mdb", media_type="application/octet-stream")

@app.post("/acceptdb")
async def accept_db(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if 2 not in current_user["role"]:
        raise HTTPException(status_code=403, detail="Only admin users can accept the database.")
    db_file = os.path.join(DATABASE_PATH, "data.mdb")
    normalized_db_file = os.path.normpath(db_file)
    try:
        global LMDB_ENV
        if LMDB_ENV is not None:
            LMDB_ENV.close()
        content = await file.read()
        temp_file = normalized_db_file + ".tmp"
        with open(temp_file, "wb") as f:
            f.write(content)
        os.replace(temp_file, normalized_db_file)
        init_lmdb()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to replace the database file: {str(e)}")
    return {"detail": "Database replaced successfully."}

@app.post("/clearfiles")
async def clear_files(current_user: dict = Depends(get_current_user)):
    try:
        for root, dirs, files in os.walk(UPLOAD_FOLDER):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {str(e)}")
    return {"detail": "All files cleared successfully."}

# --------------------------------------------------
# 14. Application Startup
# --------------------------------------------------
@app.on_event("startup")
async def on_startup():
    init_lmdb()

# --------------------------------------------------
# ADMIN SECTION (Admin Router)
# --------------------------------------------------
def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    if 2 not in current_user["role"]:
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return current_user

class AdminEditUser(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    name: Optional[str] = None
    role: Optional[List[int]] = None

class AdminCreateUser(BaseModel):
    username: str
    password: str
    name: str
    role: List[int]

admin_router = APIRouter(prefix="/admin", tags=["admin"])

@admin_router.get("/total-requests")
async def total_requests(admin: dict = Depends(get_admin_user)):
    total = len(list_all_requests())
    return {"total_requests": total}

@admin_router.get("/pending-requests")
async def pending_requests(admin: dict = Depends(get_admin_user)):
    all_requests = list_all_requests()
    pending = [r for r in all_requests if r["status"] in ("NEW", "IN_PROGRESS")]
    return {"total_pending_requests": len(pending)}

@admin_router.get("/users", response_model=List[UserResponse])
async def admin_view_all_users(admin: dict = Depends(get_admin_user)):
    return list_all_users()

@admin_router.put("/users/{user_id}", response_model=UserResponse)
async def admin_edit_user(user_id: int, user_edit: AdminEditUser, admin: dict = Depends(get_admin_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user_edit.username:
        user["username"] = user_edit.username
    if user_edit.password:
        user["hashed_password"] = get_password_hash(user_edit.password)
    if user_edit.name:
        user["name"] = user_edit.name
    if user_edit.role:
        user["role"] = user_edit.role
    update_user(user)
    return user

@admin_router.post("/users", response_model=UserResponse)
async def admin_create_user(user_data: AdminCreateUser, admin: dict = Depends(get_admin_user)):
    if get_user_by_username(user_data.username):
        raise HTTPException(status_code=400, detail="User with this username already exists.")
    new_user = {
        "username": user_data.username,
        "name": user_data.name,
        "role": user_data.role,
        "hashed_password": get_password_hash(user_data.password)
    }
    created_user = insert_user(new_user)
    return created_user

@admin_router.delete("/users/{user_id}")
async def admin_delete_user(user_id: int, admin: dict = Depends(get_admin_user)):
    with LMDB_ENV.begin(write=True, db=DB_USERS) as txn:
        key = str(user_id).encode()
        if not txn.get(key, db=DB_USERS):
            raise HTTPException(status_code=404, detail="User not found")
        txn.delete(key, db=DB_USERS)
    return {"detail": f"User {user_id} deleted successfully."}

@admin_router.get("/users/pending-requests")
async def pending_requests_per_user(admin: dict = Depends(get_admin_user)):
    users = list_all_users()
    all_requests = list_all_requests()
    results = []
    for user in users:
        pending = [r for r in all_requests if r["initiator_id"] == user["id"] and r["status"] in ("NEW", "IN_PROGRESS")]
        results.append({"user_id": user["id"], "pending_requests": len(pending)})
    return results

@admin_router.post("/requests/{request_id}/approve")
async def admin_approve_request(request_id: int, admin: dict = Depends(get_admin_user)):
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    current_time_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    req["status"] = "Approved by ADMIN"
    req["last_action"] = f"Approved by ADMIN at {current_time_str}"
    req["updated_at"] = current_time_str
    update_request(req)
    return {"detail": f"Request {request_id} approved by ADMIN."}

@admin_router.get("/users/{user_id}/files")
async def admin_view_user_files(user_id: int, admin: dict = Depends(get_admin_user)):
    user_requests = [r for r in list_all_requests() if r["initiator_id"] == user_id]
    files = []
    for req in user_requests:
        if "files" in req and isinstance(req["files"], list):
            files.extend(req["files"])
    return {"user_id": user_id, "files": files}

@admin_router.delete("/requests/{request_id}/files")
async def admin_delete_request_file(request_id: int, file_url: str, admin: dict = Depends(get_admin_user)):
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if "files" not in req or not isinstance(req["files"], list):
        raise HTTPException(status_code=404, detail="No files found for this request")

    original_files = req["files"]
    updated_files = [f for f in original_files if f.get("file_url") != file_url]
    if len(updated_files) == len(original_files):
        raise HTTPException(status_code=404, detail="File not found in the request")

    req["files"] = updated_files
    update_request(req)

    file_path = file_url.lstrip("/")
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"detail": f"File {file_url} deleted from request {request_id}."}

@admin_router.post("/requests/{request_id}/files")
async def admin_add_files_to_request(
    request_id: int,
    files: Optional[List[UploadFile]] = File(None),
    admin: dict = Depends(get_admin_user)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    file_records = req.get("files", [])
    for file in files:
        original_filename = file.filename or "unnamed_file"
        sanitized_filename = original_filename.replace(" ", "_")
        timestamp = datetime.now(IST).strftime('%Y%m%d%H%M%S')
        new_filename = f"{request_id}_{timestamp}_{sanitized_filename}"
        file_location = os.path.join(UPLOAD_FOLDER, new_filename)
        file.file.seek(0)
        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)
        file_record = {"file_url": f"/files/{new_filename}", "file_display_name": original_filename}
        file_records.append(file_record)

    req["files"] = file_records
    update_request(req)
    return {"detail": f"Files added to request {request_id}.", "files": file_records}

@admin_router.post("/requests/{request_id}/comments")
async def admin_add_comment(
    request_id: int,
    comment: str = Form(...),
    admin: dict = Depends(get_admin_user)
):
    req = get_request_by_id(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if "admin_comment" in req and req["admin_comment"]:
        req["admin_comment"] += f" | {comment}"
    else:
        req["admin_comment"] = comment

    current_time_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    req["last_action"] = f"Admin comment added at {current_time_str}"
    req["updated_at"] = current_time_str
    update_request(req)
    return {"detail": f"Comment added to request {request_id}.", "admin_comment": req["admin_comment"]}

app.include_router(admin_router)
