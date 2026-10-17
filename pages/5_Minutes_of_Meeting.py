import streamlit as st

from services import actions_service, app_init, db, mom_service
from services.ui import pick_meeting

app_init.init("Minutes of Meeting")
st.title("Minutes of Meeting")
st.caption("Drafts are generated only when you ask. Calendar files and email drafts are prepared for you: the app never sends anything or writes to a calendar by itself.")

mid = pick_meeting("mom_meeting")
if mid is None:
    st.stop()
m = db.get_meeting(mid)
key = f"mom_text_{mid}"

c1, c2 = st.columns([2, 3])
include_unconfirmed = c1.toggle("Include unconfirmed items", value=True,
                                help="Off = only items you confirmed on the Insights page.")
if c2.button("Generate MoM draft", type="primary"):
    st.session_state[key] = mom_service.build_markdown(mid, include_unconfirmed)
if key not in st.session_state:
    saved = mom_service.latest_draft(mid)
    if saved:
        st.session_state[key] = saved

if key not in st.session_state:
    st.info("Click **Generate MoM draft** to create an editable draft.")
    st.stop()

edited = st.text_area("Edit the draft (Markdown)", st.session_state[key], height=380)
st.session_state[key] = edited
b1, b2, b3 = st.columns(3)
if b1.button("Save draft"):
    mom_service.save_draft(mid, edited)
    db.audit("mom_saved", mid)
    st.toast("Draft saved")
b2.download_button("Download .md", edited, file_name=f"MoM_{m['title'].replace(' ', '_')}.md", mime="text/markdown")
b3.download_button("Download .docx", mom_service.to_docx_bytes(edited), file_name=f"MoM_{m['title'].replace(' ', '_')}.docx",
                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

with st.expander("Preview"):
    st.markdown(edited)

st.divider()
st.subheader("Calendar proposals")
acts = db.rows("SELECT * FROM action_items WHERE meeting_id=? AND deadline_date IS NOT NULL AND status!='rejected' ORDER BY deadline_date", (mid,))
if not acts:
    st.write("No action items with a resolved deadline.")
for a in acts:
    pid = actions_service.propose_event(mid, a["id"])
    with st.container(border=True):
        st.markdown(f"**{a['task']}** · {a['owner'] or 'unassigned'} · due **{a['deadline_date']}**")
        ok = st.checkbox("I confirm this event is correct", key=f"cal_ok_{a['id']}")
        if ok:
            st.download_button("Download .ics", actions_service.build_ics(pid), file_name=f"action_{a['id']}.ics",
                               mime="text/calendar", key=f"ics_{a['id']}",
                               on_click=actions_service.confirm_event, args=(pid,))
        else:
            st.caption("Tick the box to enable the download. Import the file into your calendar yourself.")

st.divider()
st.subheader("Email draft")
raw = st.text_input("Recipients (comma separated)", key="mom_recipients")
subject = st.text_input("Subject", value=f"Minutes of meeting: {m['title']} ({m['meeting_date']})")
good, bad = actions_service.parse_recipients(raw)
if bad:
    st.error("Not valid email addresses: " + ", ".join(bad))
confirm = st.checkbox("I have reviewed the minutes and the recipient list")
if st.button("Prepare email draft", disabled=not (good and not bad and confirm)):
    d = actions_service.prepare_email(mid, good, subject, edited)
    st.link_button("Open in my email app", d["mailto"])
    st.download_button("Download .eml draft", d["eml"], file_name="minutes.eml", mime="message/rfc822")
    st.caption("The email is not sent by this app." + (" The mailto body was shortened: use the .eml for the full text." if d["truncated"] else ""))
