import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Avatar,
  Card,
  CardContent,
  Grid,
  IconButton,
  Button,
  TextField,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  Tab,
  Tabs,
  Badge,
  Divider,
} from '@mui/material';
import {
  Edit as EditIcon,
  PhotoCamera as PhotoCameraIcon,
  Email as EmailIcon,
  Phone as PhoneIcon,
  LocationOn as LocationIcon,
  Work as WorkIcon,
  EmojiEvents as TrophyIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  Share as ShareIcon,
  Image as ImageIcon,
  QuestionAnswer as QuestionAnswerIcon,
  BubbleChart as BubbleChartIcon,
  Biotech as BiotechIcon,
  CheckCircle as CheckCircleIcon,
  Lock as LockIcon,
  Notifications as NotificationsIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';

const Profile = () => {
  const [editMode, setEditMode] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);
  const [openAvatarDialog, setOpenAvatarDialog] = useState(false);
  
  const [userData, setUserData] = useState({
    name: 'Dr. Rajesh Kumar',
    title: 'Marine Biologist & Aquaculture Expert',
    email: 'rajesh.kumar@meenasetu.ai',
    phone: '+91 98765 43210',
    location: 'Patna, Bihar, India',
    organization: 'Central Institute of Fisheries Education',
    bio: 'Passionate about sustainable aquaculture and fish biodiversity. Working towards improving fisheries management in West Bengal and Bihar.',
    joinDate: 'January 2024',
    avatar: '🐠'
  });

  const stats = {
    queriesAsked: 247,
    fishIdentified: 89,
    diseasesDetected: 34,
    visualizationsCreated: 56,
    documentsUploaded: 23,
    expertiseLevel: 85,
  };

  const achievements = [
    { title: 'First Query', icon: '🎯', desc: 'Asked your first question', unlocked: true },
    { title: 'Fish Expert', icon: '🐟', desc: '50+ species identified', unlocked: true },
    { title: 'Disease Detective', icon: '🔬', desc: '25+ diseases detected', unlocked: true },
    { title: 'Data Visualizer', icon: '📊', desc: 'Create 100 visualizations', unlocked: false },
    { title: 'Knowledge Contributor', icon: '📚', desc: 'Upload 50 documents', unlocked: false },
    { title: 'AI Master', icon: '🤖', desc: 'Complete 500 queries', unlocked: false },
  ];

  const recentActivity = [
    { action: 'Identified Rohu fish species', time: '2 hours ago', icon: '🐠', color: '#4caf50' },
    { action: 'Detected Fin Rot disease', time: '5 hours ago', icon: '🏥', color: '#f44336' },
    { action: 'Created population bar chart', time: '1 day ago', icon: '📊', color: '#2196f3' },
    { action: 'Asked about aquaculture practices', time: '2 days ago', icon: '💬', color: '#ff9800' },
    { action: 'Uploaded research document', time: '3 days ago', icon: '📄', color: '#9c27b0' },
  ];

  const favoriteSpecies = ['Rohu', 'Catla', 'Mrigal', 'Hilsa', 'Tilapia'];

  const avatarOptions = ['🐠', '🐟', '🐡', '🦈', '🐙', '🦀', '🦞', '🐚'];

  const handleEditToggle = () => {
    setEditMode(!editMode);
  };

  const handleInputChange = (field, value) => {
    setUserData({ ...userData, [field]: value });
  };

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleAvatarSelect = (emoji) => {
    setUserData({ ...userData, avatar: emoji });
    setOpenAvatarDialog(false);
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f7fa' }}>
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Header Section with Gradient Cover */}
        <Paper
          elevation={0}
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: 4,
            p: { xs: 3, md: 4 },
            mb: 3,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Decorative Background Elements */}
          <Box
            sx={{
              position: 'absolute',
              top: -50,
              right: -50,
              width: 200,
              height: 200,
              borderRadius: '50%',
              background: 'rgba(255,255,255,0.1)',
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              bottom: -30,
              left: -30,
              width: 150,
              height: 150,
              borderRadius: '50%',
              background: 'rgba(255,255,255,0.1)',
            }}
          />
          
          <Grid container spacing={3} alignItems="center" sx={{ position: 'relative', zIndex: 1 }}>
            <Grid item>
              <Badge
                overlap="circular"
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                badgeContent={
                  <IconButton
                    size="small"
                    sx={{
                      bgcolor: 'white',
                      boxShadow: 2,
                      '&:hover': { bgcolor: 'grey.100' },
                    }}
                    onClick={() => setOpenAvatarDialog(true)}
                  >
                    <PhotoCameraIcon fontSize="small" color="primary" />
                  </IconButton>
                }
              >
                <Avatar
                  sx={{
                    width: { xs: 100, md: 130 },
                    height: { xs: 100, md: 130 },
                    fontSize: { xs: '50px', md: '65px' },
                    bgcolor: 'rgba(255,255,255,0.3)',
                    border: '5px solid white',
                    boxShadow: 3,
                  }}
                >
                  {userData.avatar}
                </Avatar>
              </Badge>
            </Grid>
            
            <Grid item xs={12} md>
              <Box sx={{ color: 'white' }}>
                {editMode ? (
                  <TextField
                    fullWidth
                    value={userData.name}
                    onChange={(e) => handleInputChange('name', e.target.value)}
                    sx={{ 
                      mb: 1, 
                      bgcolor: 'white', 
                      borderRadius: 2,
                      '& .MuiOutlinedInput-root': { borderRadius: 2 }
                    }}
                  />
                ) : (
                  <Typography variant="h3" fontWeight="bold" gutterBottom sx={{ fontSize: { xs: '2rem', md: '3rem' } }}>
                    {userData.name}
                  </Typography>
                )}
                
                {editMode ? (
                  <TextField
                    fullWidth
                    value={userData.title}
                    onChange={(e) => handleInputChange('title', e.target.value)}
                    sx={{ 
                      bgcolor: 'white', 
                      borderRadius: 2,
                      '& .MuiOutlinedInput-root': { borderRadius: 2 }
                    }}
                  />
                ) : (
                  <Typography variant="h6" sx={{ opacity: 0.95, mb: 2 }}>
                    {userData.title}
                  </Typography>
                )}
                
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
                  <Chip
                    icon={<LocationIcon sx={{ color: 'white !important' }} />}
                    label={userData.location}
                    sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', backdropFilter: 'blur(10px)' }}
                  />
                  <Chip
                    icon={<WorkIcon sx={{ color: 'white !important' }} />}
                    label={userData.organization}
                    sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', backdropFilter: 'blur(10px)' }}
                  />
                  <Chip
                    label={`Member since ${userData.joinDate}`}
                    sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', backdropFilter: 'blur(10px)' }}
                  />
                </Box>
              </Box>
            </Grid>
            
            <Grid item>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  variant="contained"
                  startIcon={editMode ? <SettingsIcon /> : <EditIcon />}
                  onClick={handleEditToggle}
                  sx={{
                    bgcolor: 'white',
                    color: 'primary.main',
                    fontWeight: 'bold',
                    boxShadow: 2,
                    '&:hover': { bgcolor: 'grey.100', transform: 'translateY(-2px)', boxShadow: 4 },
                    transition: 'all 0.3s',
                  }}
                >
                  {editMode ? 'Save Profile' : 'Edit Profile'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ShareIcon />}
                  sx={{
                    borderColor: 'white',
                    color: 'white',
                    fontWeight: 'bold',
                    '&:hover': { 
                      borderColor: 'white', 
                      bgcolor: 'rgba(255,255,255,0.15)',
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s',
                  }}
                >
                  Share Profile
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Stats Grid with Animated Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {[
            { label: 'Queries Asked', value: stats.queriesAsked, icon: <QuestionAnswerIcon />, color: '#3f51b5', gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' },
            { label: 'Fish Identified', value: stats.fishIdentified, icon: <ImageIcon />, color: '#4caf50', gradient: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)' },
            { label: 'Diseases Detected', value: stats.diseasesDetected, icon: <BiotechIcon />, color: '#f44336', gradient: 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)' },
            { label: 'Visualizations', value: stats.visualizationsCreated, icon: <AssessmentIcon />, color: '#ff9800', gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' },
            { label: 'Documents', value: stats.documentsUploaded, icon: <TimelineIcon />, color: '#9c27b0', gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' },
            { label: 'Expertise Level', value: `${stats.expertiseLevel}%`, icon: <TrophyIcon />, color: '#ffb300', gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)' },
          ].map((stat, index) => (
            <Grid item xs={6} sm={4} md={2} key={index}>
              <Card
                sx={{
                  textAlign: 'center',
                  p: 2,
                  background: stat.gradient,
                  color: 'white',
                  height: '100%',
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: 6,
                  },
                }}
              >
                <Box sx={{ fontSize: '2.5rem', mb: 1 }}>
                  {stat.icon}
                </Box>
                <Typography variant="h4" fontWeight="bold">
                  {stat.value}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9, mt: 0.5 }}>
                  {stat.label}
                </Typography>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Tabs Section */}
        <Paper elevation={0} sx={{ borderRadius: 3, mb: 3 }}>
          <Tabs
            value={selectedTab}
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{
              borderBottom: 1,
              borderColor: 'divider',
              '& .MuiTab-root': {
                fontWeight: 'bold',
                fontSize: '1rem',
              },
            }}
          >
            <Tab label="Overview" icon={<AssessmentIcon />} iconPosition="start" />
            <Tab label="Activity" icon={<TimelineIcon />} iconPosition="start" />
            <Tab label="Achievements" icon={<TrophyIcon />} iconPosition="start" />
            <Tab label="Settings" icon={<SettingsIcon />} iconPosition="start" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            {/* Overview Tab */}
            {selectedTab === 0 && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <Card sx={{ mb: 3, borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        About Me
                      </Typography>
                      {editMode ? (
                        <TextField
                          fullWidth
                          multiline
                          rows={4}
                          value={userData.bio}
                          onChange={(e) => handleInputChange('bio', e.target.value)}
                        />
                      ) : (
                        <Typography color="text.secondary">
                          {userData.bio}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>

                  <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        Contact Information
                      </Typography>
                      <List>
                        <ListItem>
                          <ListItemIcon>
                            <EmailIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText 
                            primary="Email" 
                            secondary={userData.email}
                          />
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemIcon>
                            <PhoneIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText 
                            primary="Phone" 
                            secondary={userData.phone}
                          />
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemIcon>
                            <LocationIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText 
                            primary="Location" 
                            secondary={userData.location}
                          />
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Card sx={{ mb: 3, borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        Expertise Progress
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            Level {Math.floor(stats.expertiseLevel / 20)}
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {stats.expertiseLevel}%
                          </Typography>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={stats.expertiseLevel} 
                          sx={{ 
                            height: 10, 
                            borderRadius: 5,
                            '& .MuiLinearProgress-bar': {
                              background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                            }
                          }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          {100 - stats.expertiseLevel}% to next level
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>

                  <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        Favorite Species
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                        {favoriteSpecies.map((species, index) => (
                          <Chip
                            key={index}
                            label={species}
                            color="primary"
                            variant="outlined"
                            sx={{ fontWeight: 'bold' }}
                          />
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}

            {/* Activity Tab */}
            {selectedTab === 1 && (
              <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
                <CardContent>
                  <Typography variant="h6" fontWeight="bold" gutterBottom>
                    Recent Activity
                  </Typography>
                  <List>
                    {recentActivity.map((activity, index) => (
                      <React.Fragment key={index}>
                        <ListItem
                          sx={{
                            borderRadius: 2,
                            mb: 1,
                            '&:hover': { bgcolor: 'action.hover' },
                            transition: 'all 0.2s',
                          }}
                        >
                          <ListItemIcon>
                            <Avatar sx={{ bgcolor: activity.color, width: 40, height: 40 }}>
                              {activity.icon}
                            </Avatar>
                          </ListItemIcon>
                          <ListItemText
                            primary={activity.action}
                            secondary={activity.time}
                            primaryTypographyProps={{ fontWeight: 'medium' }}
                          />
                        </ListItem>
                        {index < recentActivity.length - 1 && <Divider />}
                      </React.Fragment>
                    ))}
                  </List>
                </CardContent>
              </Card>
            )}

            {/* Achievements Tab */}
            {selectedTab === 2 && (
              <Grid container spacing={2}>
                {achievements.map((achievement, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card
                      sx={{
                        textAlign: 'center',
                        p: 3,
                        borderRadius: 3,
                        boxShadow: 2,
                        opacity: achievement.unlocked ? 1 : 0.5,
                        background: achievement.unlocked 
                          ? 'linear-gradient(135deg, #667eea20 0%, #764ba220 100%)'
                          : 'transparent',
                        border: achievement.unlocked ? '2px solid #667eea' : '2px solid #ddd',
                        transition: 'all 0.3s',
                        '&:hover': {
                          transform: achievement.unlocked ? 'scale(1.05)' : 'none',
                          boxShadow: achievement.unlocked ? 4 : 2,
                        },
                      }}
                    >
                      <Box sx={{ fontSize: '4rem', mb: 2 }}>
                        {achievement.unlocked ? achievement.icon : '🔒'}
                      </Box>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        {achievement.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {achievement.desc}
                      </Typography>
                      {achievement.unlocked ? (
                        <Chip
                          icon={<CheckCircleIcon />}
                          label="Unlocked"
                          color="success"
                          size="small"
                          sx={{ mt: 1 }}
                        />
                      ) : (
                        <Chip
                          icon={<LockIcon />}
                          label="Locked"
                          size="small"
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}

            {/* Settings Tab */}
            {selectedTab === 3 && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Card sx={{ borderRadius: 2, boxShadow: 2, mb: 3 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        <NotificationsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Notifications
                      </Typography>
                      <List>
                        <ListItem>
                          <ListItemText primary="Email Notifications" secondary="Receive updates via email" />
                          <Button variant="outlined" size="small">Enable</Button>
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemText primary="Activity Alerts" secondary="Get notified about new activity" />
                          <Button variant="contained" size="small">Enabled</Button>
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card sx={{ borderRadius: 2, boxShadow: 2, mb: 3 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Security
                      </Typography>
                      <List>
                        <ListItem>
                          <ListItemText primary="Change Password" secondary="Update your password" />
                          <Button variant="outlined" size="small">Change</Button>
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemText primary="Two-Factor Auth" secondary="Add extra security" />
                          <Button variant="outlined" size="small">Setup</Button>
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
          </Box>
        </Paper>
      </Container>

      {/* Avatar Selection Dialog */}
      <Dialog open={openAvatarDialog} onClose={() => setOpenAvatarDialog(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Choose Your Avatar</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            {avatarOptions.map((emoji, index) => (
              <Grid item xs={3} key={index}>
                <IconButton
                  onClick={() => handleAvatarSelect(emoji)}
                  sx={{
                    fontSize: '3rem',
                    width: '100%',
                    aspectRatio: '1',
                    border: userData.avatar === emoji ? '3px solid #667eea' : '2px solid #ddd',
                    borderRadius: 2,
                    '&:hover': {
                      bgcolor: 'action.hover',
                      transform: 'scale(1.1)',
                    },
                    transition: 'all 0.2s',
                  }}
                >
                  {emoji}
                </IconButton>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAvatarDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Profile;