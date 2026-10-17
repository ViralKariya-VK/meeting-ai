"""Calendar proposals (.ics) and email drafts. The app never sends mail or writes to a calendar:
it prepares a file/draft, and a person must explicitly confirm and perform the external step."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

from . import db


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def make_uid(meeting_id: int, action_id: int, title: str, event_date: str) -> str:
    h = hashlib.sha1(f"{meeting_id}|{action_id}|{title}|{event_date}".encode()).hexdigest()[:20]
    return f"{h}@meeting-ai.demo"


def propose_event(meeting_id: int, action_id: int) -> Optional[int]:
    """Create (idempotently) a calendar proposal for an action item with a resolved deadline."""
    a = db.one("SELECT * FROM action_items WHERE id=? AND meeting_id=?", (action_id, meeting_id))
    if a is None or not a["deadline_date"]:
        return None
    title = f"{a['task']}" + (f" ({a['owner']})" if a["owner"] else "")
    uid = make_uid(meeting_id, action_id, title, a["deadline_date"])
    ex = db.one("SELECT id FROM calendar_proposals WHERE uid=?", (uid,))
    if ex:
        return ex["id"]
    pid = db.execute(
        "INSERT INTO calendar_proposals(meeting_id, action_item_id, title, event_date, description, uid) VALUES (?,?,?,?,?,?)",
        (meeting_id, action_id, title, a["deadline_date"], a["evidence_quote"] or "", uid))
    db.audit("calendar_proposed", meeting_id, f"proposal={pid}")
    return pid


def build_ics(proposal_id: int) -> str:
    p = db.one("SELECT * FROM calendar_proposals WHERE id=?", (proposal_id,))
    if p is None:
        raise ValueError("Unknown proposal")
    d = date.fromisoformat(p["event_date"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Meeting AI Demo//EN", "CALSCALE:GREGORIAN", "METHOD:PUBLISH",
        "BEGIN:VEVENT", f"UID:{p['uid']}", f"DTSTAMP:{stamp}",
        f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}", f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}",
        f"SUMMARY:{_esc(p['title'])}", f"DESCRIPTION:{_esc(p['description'] or '')}", "END:VEVENT", "END:VCALENDAR", ""])


def confirm_event(proposal_id: int) -> None:
    """Record that the user confirmed (and downloaded) the proposal. Idempotent."""
    p = db.one("SELECT meeting_id, status FROM calendar_proposals WHERE id=?", (proposal_id,))
    if p and p["status"] != "confirmed":
        db.execute("UPDATE calendar_proposals SET status='confirmed' WHERE id=?", (proposal_id,))
        db.audit("calendar_confirmed", p["meeting_id"], f"proposal={proposal_id}")


_EMAIL = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")


def parse_recipients(raw: str) -> tuple[list[str], list[str]]:
    items = [x.strip() for x in re.split(r"[,;\n]", raw) if x.strip()]
    return [i for i in items if _EMAIL.match(i)], [i for i in items if not _EMAIL.match(i)]


def prepare_email(meeting_id: int, recipients: list[str], subject: str, body_md: str) -> dict:
    """Build a mailto link and .eml text. Logs that a draft was prepared; nothing is transmitted."""
    body = re.sub(r"[*_#|]", "", body_md)
    body = re.sub(r"\n{3,}", "\n\n", body)
    mailto = f"mailto:{','.join(recipients)}?subject={quote(subject)}&body={quote(body[:1500])}"
    eml = (f"To: {', '.join(recipients)}\r\nSubject: {subject}\r\nContent-Type: text/plain; charset=utf-8\r\n"
           f"X-Unsent: 1\r\n\r\n{body}")
    db.execute("INSERT INTO email_deliveries(meeting_id, recipients, subject, status) VALUES (?,?,?,?)",
               (meeting_id, ", ".join(recipients), subject, "draft_prepared"))
    db.audit("email_draft_prepared", meeting_id, f"{len(recipients)} recipient(s)")
    return {"mailto": mailto, "eml": eml, "truncated": len(body) > 1500}
