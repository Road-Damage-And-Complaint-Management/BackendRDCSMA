# Road Damage Detection & Complaint Management System  

This project is a web-based platform that enables users to report road damages using AI-powered image analysis and automatic geo-tagging. The system provides an admin dashboard for municipal authorities to manage complaints and prioritize road repairs efficiently.  

## Features  

### AI-Based Road Damage Detection  
Users upload road images, and the system detects cracks using the YOLOv8 AI model. It identifies the crack type and depth for accurate damage assessment.  

### Automatic Geo-Tagging  
The system extracts GPS coordinates from uploaded images and converts them into real-world locations.  

### Admin Dashboard for Complaint Management  
The admin dashboard includes authentication for secure access. A dynamic report table allows sorting based on depth value, where higher depth indicates higher priority, and multiple user confirmations for the same location increase priority. Real-time status updates are provided for complaints.  

### User Dashboard  
Users can view past reports and track their status. The interface is designed for an intuitive experience when submitting complaints.  

### Priority Calculation System  

#### Depth-Based Priority  
Reports with greater crack depth values are ranked higher.  

#### Multiple User Reports  
If two or more users report the same location and crack type, the priority increases.  

#### Sorting Order  
Priority score is the primary sorting factor, followed by depth value in case of equal scores.  


## Technologies Used  

### Frontend  
React.js, HTML, CSS, Bootstrap or Tailwind  

### Backend  
Node.js, Flask, PostgreSQL or MySQL, PyTorch, TensorFlow  
