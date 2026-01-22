import React, { useState, useEffect } from 'react';
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
  CircularProgress,
  Alert,
  Snackbar,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
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
  History as HistoryIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  CalendarToday as CalendarIcon,
  FiberManualRecord as DotIcon,
} from '@mui/icons-material';

const API_BASE = 'http://localhost:8000';

const EnhancedProfile = () => {
  const [editMode, setEditMode] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);
  const [openAvatarDialog, setOpenAvatarDialog] = useState(false);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  
  // Real data from API
  const [apiStats, setApiStats] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [recentQueries, setRecentQueries] = useState([]);
  const [speciesList, setSpeciesList] = useState([]);
  const [diseasesList, setDiseasesList] = useState([]);
  
  const [userData, setUserData] = useState({
    name: 'Dr. Rajesh Kumar',
    title: 'Marine Biologist & Aquaculture Expert',
    email: 'rajesh.kumar@meenasetu.ai',
    phone: '+91 98765 43210',
    location: 'Patna, Bihar, India',
    organization: 'Central Institute of Fisheries Education',
    bio: 'Passionate about sustainable aquaculture and fish biodiversity. Working towards improving fisheries management in West Bengal and Bihar.',
    joinDate: 'January 2024',
    avatar: null,
    avatarEmoji: '🐠'
  });

  const [stats, setStats] = useState({
    queriesAsked: 0,
    fishIdentified: 0,
    diseasesDetected: 0,
    visualizationsCreated: 0,
    documentsUploaded: 0,
    expertiseLevel: 0,
  });

  const achievements = [
    { title: 'First Query', icon: '🎯', desc: 'Asked your first question', threshold: 1, field: 'queriesAsked' },
    { title: 'Fish Expert', icon: '🐟', desc: '50+ species identified', threshold: 50, field: 'fishIdentified' },
    { title: 'Disease Detective', icon: '🔬', desc: '25+ diseases detected', threshold: 25, field: 'diseasesDetected' },
    { title: 'Data Visualizer', icon: '📊', desc: 'Create 50 visualizations', threshold: 50, field: 'visualizationsCreated' },
    { title: 'Knowledge Contributor', icon: '📚', desc: 'Upload 50 documents', threshold: 50, field: 'documentsUploaded' },
    { title: 'AI Master', icon: '🤖', desc: 'Complete 500 queries', threshold: 500, field: 'queriesAsked' },
  ];

  const avatarOptions = ['🐠', '🐟', '🐡', '🦈', '🐙', '🦀', '🦞', '🐚', '🐋', '🦑', '🐬', '🦭'];

  // Fetch data from API
  useEffect(() => {
    fetchAPIData();
  }, []);

  const fetchAPIData = async () => {
    setLoading(true);
    try {
      // Fetch statistics
      const statsRes = await fetch(`${API_BASE}/stats`);
      const statsData = await statsRes.json();
      
      if (statsData.statistics) {
        setApiStats(statsData);
        
        // Calculate stats from API data
        const queries = statsData.statistics.session_info?.queries_processed || 0;
        const docs = statsData.statistics.database_stats?.total_documents || 0;
        
        setStats(prev => ({
          ...prev,
          queriesAsked: queries,
          documentsUploaded: docs,
          expertiseLevel: Math.min(Math.floor((queries / 10) * 100), 100),
        }));
      }

      // Fetch conversation history
      const historyRes = await fetch(`${API_BASE}/conversation/history?limit=50`);
      const historyData = await historyRes.json();
      if (historyData.history) {
        setConversationHistory(historyData.history);
        
        // Extract recent queries
        const queries = historyData.history
          .filter(item => item.role === 'user')
          .slice(-10)
          .reverse();
        setRecentQueries(queries);
      }

      // Fetch species list
      const speciesRes = await fetch(`${API_BASE}/docs/species-list`);
      const speciesData = await speciesRes.json();
      if (speciesData.species) {
        setSpeciesList(speciesData.species);
      }

      // Fetch diseases list
      const diseasesRes = await fetch(`${API_BASE}/docs/diseases`);
      const diseasesData = await diseasesRes.json();
      if (diseasesData.detectable_diseases) {
        setDiseasesList(diseasesData.detectable_diseases);
      }

      setSnackbar({ open: true, message: 'Profile data loaded successfully!', severity: 'success' });
    } catch (error) {
      console.error('Error fetching data:', error);
      setSnackbar({ open: true, message: 'Failed to load some data. Using cached values.', severity: 'warning' });
    } finally {
      setLoading(false);
    }
  };

  const handleEditToggle = () => {
    if (editMode) {
      setSnackbar({ open: true, message: 'Profile saved successfully!', severity: 'success' });
    }
    setEditMode(!editMode);
  };

  const handleInputChange = (field, value) => {
    setUserData({ ...userData, [field]: value });
  };

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleAvatarSelect = (emoji) => {
    setUserData({ ...userData, avatarEmoji: emoji });
    setOpenAvatarDialog(false);
    setSnackbar({ open: true, message: 'Avatar updated!', severity: 'success' });
  };

  const handleAvatarUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadingAvatar(true);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setUserData({ ...userData, avatar: e.target.result, avatarEmoji: null });
      setUploadingAvatar(false);
      setSnackbar({ open: true, message: 'Profile picture uploaded!', severity: 'success' });
    };
    reader.readAsDataURL(file);
  };

  const handleExportHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversation/export`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `meenasetu_history_${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      setSnackbar({ open: true, message: 'History exported successfully!', severity: 'success' });
    } catch (error) {
      setSnackbar({ open: true, message: 'Export failed', severity: 'error' });
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Recently';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f7fa' }}>
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Snackbar */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={4000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
            {snackbar.message}
          </Alert>
        </Snackbar>

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
                  <>
                    <input
                      accept="image/*"
                      style={{ display: 'none' }}
                      id="avatar-upload"
                      type="file"
                      onChange={handleAvatarUpload}
                    />
                    <label htmlFor="avatar-upload">
                      <IconButton
                        component="span"
                        size="small"
                        sx={{
                          bgcolor: 'white',
                          boxShadow: 2,
                          '&:hover': { bgcolor: 'grey.100' },
                        }}
                      >
                        {uploadingAvatar ? <CircularProgress size={16} /> : <PhotoCameraIcon fontSize="small" color="primary" />}
                      </IconButton>
                    </label>
                  </>
                }
              >
                <Avatar
                  src={userData.avatar}
                  sx={{
                    width: { xs: 100, md: 130 },
                    height: { xs: 100, md: 130 },
                    fontSize: { xs: '50px', md: '65px' },
                    bgcolor: 'rgba(255,255,255,0.3)',
                    border: '5px solid white',
                    boxShadow: 3,
                    cursor: 'pointer',
                  }}
                  onClick={() => !userData.avatar && setOpenAvatarDialog(true)}
                >
                  {!userData.avatar && userData.avatarEmoji}
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
                  {loading && (
                    <Chip
                      icon={<CircularProgress size={16} sx={{ color: 'white' }} />}
                      label="Syncing..."
                      sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white' }}
                    />
                  )}
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
                  startIcon={<RefreshIcon />}
                  onClick={fetchAPIData}
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
                  Refresh Data
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Real-Time Stats Grid */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {[
            { label: 'Queries Asked', value: stats.queriesAsked, icon: <QuestionAnswerIcon />, gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' },
            { label: 'Fish Species Known', value: speciesList.length || 31, icon: <ImageIcon />, gradient: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)' },
            { label: 'Diseases Trackable', value: diseasesList.length || 6, icon: <BiotechIcon />, gradient: 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)' },
            { label: 'Documents', value: stats.documentsUploaded, icon: <TimelineIcon />, gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' },
            { label: 'DB Records', value: apiStats?.statistics?.database_stats?.total_documents || 0, icon: <AssessmentIcon />, gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' },
            { label: 'Expertise Level', value: `${stats.expertiseLevel}%`, icon: <TrophyIcon />, gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)' },
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
            <Tab label="Query History" icon={<HistoryIcon />} iconPosition="start" />
            <Tab label="Knowledge Base" icon={<BubbleChartIcon />} iconPosition="start" />
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

                  <Card sx={{ mb: 3, borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6" fontWeight="bold">
                          Recent Activity
                        </Typography>
                        <Chip label={`${recentQueries.length} queries`} color="primary" size="small" />
                      </Box>
                      <List>
                        {recentQueries.slice(0, 5).map((query, index) => (
                          <React.Fragment key={index}>
                            <ListItem
                              sx={{
                                borderRadius: 2,
                                mb: 1,
                                bgcolor: 'grey.50',
                                '&:hover': { bgcolor: 'grey.100' },
                                transition: 'all 0.2s',
                              }}
                            >
                              <ListItemIcon>
                                <Avatar sx={{ bgcolor: '#667eea', width: 40, height: 40 }}>
                                  <QuestionAnswerIcon fontSize="small" />
                                </Avatar>
                              </ListItemIcon>
                              <ListItemText
                                primary={query.content?.substring(0, 80) + (query.content?.length > 80 ? '...' : '')}
                                secondary={formatTimestamp(query.timestamp)}
                                primaryTypographyProps={{ fontWeight: 'medium' }}
                              />
                            </ListItem>
                            {index < recentQueries.slice(0, 5).length - 1 && <Divider />}
                          </React.Fragment>
                        ))}
                        {recentQueries.length === 0 && (
                          <Typography color="text.secondary" align="center" sx={{ py: 3 }}>
                            No queries yet. Start asking questions!
                          </Typography>
                        )}
                      </List>
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
                            secondary={editMode ? (
                              <TextField
                                size="small"
                                value={userData.email}
                                onChange={(e) => handleInputChange('email', e.target.value)}
                              />
                            ) : userData.email}
                          />
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemIcon>
                            <PhoneIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText 
                            primary="Phone" 
                            secondary={editMode ? (
                              <TextField
                                size="small"
                                value={userData.phone}
                                onChange={(e) => handleInputChange('phone', e.target.value)}
                              />
                            ) : userData.phone}
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
                          {stats.queriesAsked < 500 ? `${500 - stats.queriesAsked} more queries to master level` : 'Master level achieved!'}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>

                  <Card sx={{ mb: 3, borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        System Status
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemIcon>
                            <DotIcon sx={{ color: apiStats ? 'success.main' : 'error.main' }} />
                          </ListItemIcon>
                          <ListItemText 
                            primary="API Connection"
                            secondary={apiStats ? 'Connected' : 'Disconnected'}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <DotIcon sx={{ color: speciesList.length > 0 ? 'success.main' : 'warning.main' }} />
                          </ListItemIcon>
                          <ListItemText 
                            primary="ML Models"
                            secondary={`${speciesList.length} species loaded`}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <DotIcon sx={{ color: conversationHistory.length > 0 ? 'success.main' : 'info.main' }} />
                          </ListItemIcon>
                          <ListItemText 
                            primary="Chat History"
                            secondary={`${conversationHistory.length} messages`}
                          />
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>

                  <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        Quick Stats
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                          <Typography variant="body2" color="text.secondary">Session Start</Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {apiStats?.statistics?.session_info?.start_time?.split('T')[1]?.split('.')[0] || 'N/A'}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                          <Typography variant="body2" color="text.secondary">Database Size</Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {apiStats?.statistics?.database_stats?.total_documents || 0} docs
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">ML Models</Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {apiStats?.ml_models?.species_models || 0} loaded
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}

            {/* Query History Tab */}
            {selectedTab === 1 && (
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h5" fontWeight="bold">
                    Your Query History
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<DownloadIcon />}
                    onClick={handleExportHistory}
                    sx={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    }}
                  >
                    Export History
                  </Button>
                </Box>

                <TableContainer component={Card} sx={{ borderRadius: 2, boxShadow: 2 }}>
                  <Table>
                    <TableHead>
                      <TableRow sx={{ bgcolor: 'grey.100' }}>
                        <TableCell><strong>Query</strong></TableCell>
                        <TableCell><strong>Type</strong></TableCell>
                        <TableCell><strong>Time</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {conversationHistory
                        .filter(item => item.role === 'user')
                        .slice(-20)
                        .reverse()
                        .map((item, index) => (
                          <TableRow 
                            key={index}
                            sx={{ '&:hover': { bgcolor: 'grey.50' }, cursor: 'pointer' }}
                          >
                            <TableCell>
                              <Typography variant="body2">
                                {item.content?.substring(0, 100) + (item.content?.length > 100 ? '...' : '')}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label="Text Query" 
                                size="small" 
                                color="primary" 
                                variant="outlined"
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="text.secondary">
                                {formatTimestamp(item.timestamp)}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))}
                      {conversationHistory.filter(item => item.role === 'user').length === 0 && (
                        <TableRow>
                          <TableCell colSpan={3} align="center" sx={{ py: 5 }}>
                            <Typography color="text.secondary">
                              No queries yet. Start exploring MeenaSetu AI!
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            {/* Knowledge Base Tab */}
            {selectedTab === 2 && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Card sx={{ borderRadius: 2, boxShadow: 2, height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <ImageIcon sx={{ mr: 1, color: 'primary.main' }} />
                        Fish Species Database
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {speciesList.length} species available for identification
                      </Typography>
                      <Box sx={{ maxHeight: 400, overflow: 'auto', mt: 2 }}>
                        <Grid container spacing={1}>
                          {speciesList.map((species, index) => (
                            <Grid item xs={6} key={index}>
                              <Chip
                                label={species}
                                size="small"
                                sx={{ 
                                  width: '100%',
                                  justifyContent: 'flex-start',
                                  '&:hover': { bgcolor: 'primary.light', color: 'white' }
                                }}
                              />
                            </Grid>
                          ))}
                        </Grid>
                        {speciesList.length === 0 && (
                          <Typography color="text.secondary" align="center" sx={{ py: 3 }}>
                            Loading species data...
                          </Typography>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card sx={{ borderRadius: 2, boxShadow: 2, height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                        <BiotechIcon sx={{ mr: 1, color: 'error.main' }} />
                        Disease Detection System
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {diseasesList.length} diseases can be detected
                      </Typography>
                      <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                        {diseasesList.map((disease, index) => (
                          <ListItem 
                            key={index}
                            sx={{ 
                              borderRadius: 1,
                              mb: 1,
                              bgcolor: 'error.lighter',
                              '&:hover': { bgcolor: 'error.light' }
                            }}
                          >
                            <ListItemIcon>
                              <BiotechIcon color="error" />
                            </ListItemIcon>
                            <ListItemText 
                              primary={disease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                              primaryTypographyProps={{ fontWeight: 'medium' }}
                            />
                          </ListItem>
                        ))}
                        {diseasesList.length === 0 && (
                          <Typography color="text.secondary" align="center" sx={{ py: 3 }}>
                            Loading disease data...
                          </Typography>
                        )}
                      </List>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12}>
                  <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        API System Information
                      </Typography>
                      <Grid container spacing={2} sx={{ mt: 1 }}>
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light', color: 'white' }}>
                            <Typography variant="h4" fontWeight="bold">
                              {apiStats?.statistics?.database_stats?.total_documents || 0}
                            </Typography>
                            <Typography variant="body2">Documents in DB</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light', color: 'white' }}>
                            <Typography variant="h4" fontWeight="bold">
                              {apiStats?.ml_models?.species_models || 0}
                            </Typography>
                            <Typography variant="body2">ML Models</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light', color: 'white' }}>
                            <Typography variant="h4" fontWeight="bold">
                              {speciesList.length}
                            </Typography>
                            <Typography variant="body2">Species</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light', color: 'white' }}>
                            <Typography variant="h4" fontWeight="bold">
                              {diseasesList.length}
                            </Typography>
                            <Typography variant="body2">Diseases</Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}

            {/* Achievements Tab */}
            {selectedTab === 3 && (
              <Grid container spacing={2}>
                {achievements.map((achievement, index) => {
                  const unlocked = stats[achievement.field] >= achievement.threshold;
                  const progress = Math.min((stats[achievement.field] / achievement.threshold) * 100, 100);
                  
                  return (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <Card
                        sx={{
                          textAlign: 'center',
                          p: 3,
                          borderRadius: 3,
                          boxShadow: 2,
                          opacity: unlocked ? 1 : 0.5,
                          background: unlocked 
                            ? 'linear-gradient(135deg, #667eea20 0%, #764ba220 100%)'
                            : 'transparent',
                          border: unlocked ? '2px solid #667eea' : '2px solid #ddd',
                          transition: 'all 0.3s',
                          '&:hover': {
                            transform: unlocked ? 'scale(1.05)' : 'none',
                            boxShadow: unlocked ? 4 : 2,
                          },
                        }}
                      >
                        <Box sx={{ fontSize: '4rem', mb: 2 }}>
                          {unlocked ? achievement.icon : '🔒'}
                        </Box>
                        <Typography variant="h6" fontWeight="bold" gutterBottom>
                          {achievement.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {achievement.desc}
                        </Typography>
                        {unlocked ? (
                          <Chip
                            icon={<CheckCircleIcon />}
                            label="Unlocked"
                            color="success"
                            size="small"
                            sx={{ mt: 1 }}
                          />
                        ) : (
                          <>
                            <Chip
                              icon={<LockIcon />}
                              label={`${stats[achievement.field]}/${achievement.threshold}`}
                              size="small"
                              sx={{ mt: 1, mb: 1 }}
                            />
                            <LinearProgress 
                              variant="determinate" 
                              value={progress} 
                              sx={{ mt: 1, height: 6, borderRadius: 3 }}
                            />
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                              {Math.round(progress)}% complete
                            </Typography>
                          </>
                        )}
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            )}

            {/* Settings Tab */}
            {selectedTab === 4 && (
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
                        <Divider />
                        <ListItem>
                          <ListItemText primary="Query Reminders" secondary="Daily query suggestions" />
                          <Button variant="outlined" size="small">Disabled</Button>
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>

                  <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
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
                        <Divider />
                        <ListItem>
                          <ListItemText primary="API Access" secondary="Manage API keys" />
                          <Button variant="outlined" size="small">Manage</Button>
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card sx={{ borderRadius: 2, boxShadow: 2, mb: 3 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        Data Management
                      </Typography>
                      <List>
                        <ListItem>
                          <ListItemText 
                            primary="Export All Data" 
                            secondary="Download your complete profile and history" 
                          />
                          <Button 
                            variant="outlined" 
                            size="small"
                            startIcon={<DownloadIcon />}
                            onClick={handleExportHistory}
                          >
                            Export
                          </Button>
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemText 
                            primary="Clear History" 
                            secondary="Remove all query history" 
                          />
                          <Button 
                            variant="outlined" 
                            size="small" 
                            color="warning"
                          >
                            Clear
                          </Button>
                        </ListItem>
                        <Divider />
                        <ListItem>
                          <ListItemText 
                            primary="Refresh Cache" 
                            secondary="Reload data from server" 
                          />
                          <Button 
                            variant="outlined" 
                            size="small"
                            startIcon={<RefreshIcon />}
                            onClick={fetchAPIData}
                          >
                            Refresh
                          </Button>
                        </ListItem>
                      </List>
                    </CardContent>
                  </Card>

                  <Card sx={{ borderRadius: 2, boxShadow: 2 }}>
                    <CardContent>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        Preferences
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Language</InputLabel>
                          <Select defaultValue="en" label="Language">
                            <MenuItem value="en">English</MenuItem>
                            <MenuItem value="hi">Hindi</MenuItem>
                            <MenuItem value="bn">Bengali</MenuItem>
                          </Select>
                        </FormControl>
                        
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Theme</InputLabel>
                          <Select defaultValue="light" label="Theme">
                            <MenuItem value="light">Light</MenuItem>
                            <MenuItem value="dark">Dark</MenuItem>
                            <MenuItem value="auto">Auto</MenuItem>
                          </Select>
                        </FormControl>

                        <FormControl fullWidth>
                          <InputLabel>Query Display</InputLabel>
                          <Select defaultValue="detailed" label="Query Display">
                            <MenuItem value="detailed">Detailed</MenuItem>
                            <MenuItem value="compact">Compact</MenuItem>
                            <MenuItem value="minimal">Minimal</MenuItem>
                          </Select>
                        </FormControl>
                      </Box>
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
                <Tooltip title={`Select ${emoji}`}>
                  <IconButton
                    onClick={() => handleAvatarSelect(emoji)}
                    sx={{
                      fontSize: '3rem',
                      width: '100%',
                      aspectRatio: '1',
                      border: userData.avatarEmoji === emoji ? '3px solid #667eea' : '2px solid #ddd',
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
                </Tooltip>
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

export default EnhancedProfile;