from enum import Enum


class UserRole(str, Enum):
    DOCTOR = "doctor"
    RESEARCHER = "researcher"
    HOSPITAL_ADMIN = "hospital_admin"
    SYSTEM_ADMIN = "system_admin"
