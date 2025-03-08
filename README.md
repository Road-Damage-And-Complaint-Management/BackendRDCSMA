🚧 Road Damage Detection & Complaint Management System
This project is a web-based platform that enables users to report road damages using AI-powered image analysis and automatic geo-tagging. The system provides an admin dashboard for municipal authorities to manage complaints and prioritize road repairs efficiently.

🌟 Features
🔍 AI-Based Road Damage Detection
✅ Users upload road images, and the system detects cracks using YOLOv8 AI model.
✅ Identifies crack type and depth for accurate damage assessment.

📍 Automatic Geo-Tagging
✅ Extracts GPS coordinates from uploaded images.
✅ Converts coordinates into real-world locations.

🛠 Admin Dashboard for Complaint Management
✅ Admin Authentication for secure access.
✅ Dynamic Report Table with sorting based on:

Depth value (higher depth = higher priority).
Multiple user confirmations for the same location.
✅ Real-time status updates for complaints.
👥 User Dashboard
✅ Users can view past reports and track their status.
✅ Intuitive UI for submitting complaints.

🔢 Priority Calculation System
1️⃣ Depth-Based Priority (10 Points)

Reports with greater crack depth values are ranked higher.
2️⃣ Multiple User Reports (10 Points)

If 2 or more users report the same location & crack type, priority increases.
🔝 Sorting Order:

Priority Score (Higher First)
Depth Value (Higher First, if Scores are Equal)
📷 Screenshots
(Add screenshots of the system here!)

🛠 Technologies Used
Frontend
React.js
HTML, CSS
Bootstrap / Tailwind (if applicable)
Backend
Node.js & Express.js (API Handling)
Flask (AI Model Integration)
PostgreSQL / MySQL / MongoDB (Database)
AI Model
YOLOv8 (for road damage detection)
PyTorch / TensorFlow
