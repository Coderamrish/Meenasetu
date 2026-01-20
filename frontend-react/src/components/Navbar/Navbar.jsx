import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  CircularProgress,
  Switch,
  FormControlLabel,
  Button,
  Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  DarkMode,
  LightMode,
  Person,
  Edit,
  Save,
  Waves,
  Logout,
} from '@mui/icons-material';

const Navbar = ({ onMenuClick }) => {
  const navigate = useNavigate();
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const [profileDialog, setProfileDialog] = useState(false);
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      title: 'New Data Available',
      message: 'Atlantic Ocean temperature data updated',
      created_at: new Date(),
      is_read: false,
    },
    {
      id: 2,
      title: 'Analysis Complete',
      message: 'Your salinity analysis is ready',
      created_at: new Date(Date.now() - 86400000),
      is_read: false,
    },
    {
      id: 3,
      title: 'System Update',
      message: 'Platform maintenance scheduled',
      created_at: new Date(Date.now() - 172800000),
      is_read: true,
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [profileData, setProfileData] = useState({
    full_name: 'John Doe',
    email: 'john@example.com',
    username: 'johndoe',
  });
  const [editMode, setEditMode] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  const handleNotificationMenuOpen = (event) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationMenuClose = () => {
    setNotificationAnchor(null);
  };

  const handleProfileDialogOpen = () => {
    setProfileDialog(true);
    setEditMode(false);
  };

  const handleProfileDialogClose = () => {
    setProfileDialog(false);
    setEditMode(false);
  };

  const handleProfileUpdate = async () => {
    try {
      setLoading(true);
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      console.log('Profile updated:', profileData);
      setEditMode(false);
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    console.log('Logging out...');
    // Clear any auth tokens/session data here
    navigate('/login');
  };

  const toggleDarkMode = () => {
    setDarkMode((prev) => !prev);
  };

  const markNotificationAsRead = (notificationId) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === notificationId ? { ...n, is_read: true } : n))
    );
  };

  const unreadCount = notifications.filter((n) => !n.is_read).length;

  return (
    <>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        }}
      >
        <Toolbar sx={{ minHeight: { xs: 64, sm: 70 }, px: { xs: 2, sm: 3 } }}>
          <Tooltip title="Menu" arrow>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              onClick={onMenuClick}
              edge="start"
              sx={{
                mr: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.15)',
                  transform: 'scale(1.05)',
                },
                transition: 'all 0.2s ease-in-out',
              }}
            >
              <MenuIcon />
            </IconButton>
          </Tooltip>

          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
              gap: 1.5,
              flexGrow: 1,
              '&:hover': {
                transform: 'translateY(-2px)',
              },
              transition: 'transform 0.2s ease-in-out',
            }}
            onClick={() => navigate('/')}
          >
            <Avatar
              sx={{
                bgcolor: 'rgba(255, 255, 255, 0.2)',
                width: 40,
                height: 40,
                backdropFilter: 'blur(10px)',
                border: '2px solid rgba(255, 255, 255, 0.3)',
              }}
            >
              <Waves sx={{ color: '#fff' }} />
            </Avatar>

            <Box>
              <Typography
                variant="h6"
                component="div"
                sx={{
                  fontWeight: 700,
                  letterSpacing: '-0.5px',
                  lineHeight: 1.2,
                  fontSize: { xs: '1.1rem', sm: '1.3rem' },
                  color: 'white',
                }}
              >
                MeenaSetu AI
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  opacity: 0.9,
                  fontSize: '0.7rem',
                  color: 'white',
                  display: { xs: 'none', sm: 'block' },
                }}
              >
                Aquatic Intelligence Platform
              </Typography>
            </Box>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Toggle Dark Mode" arrow>
              <IconButton
                onClick={toggleDarkMode}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                    transform: 'rotate(180deg)',
                  },
                  transition: 'all 0.3s ease-in-out',
                }}
              >
                {darkMode ? <LightMode /> : <DarkMode />}
              </IconButton>
            </Tooltip>

            <Tooltip title="Notifications" arrow>
              <IconButton
                onClick={handleNotificationMenuOpen}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                    transform: 'scale(1.05)',
                  },
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                <Badge badgeContent={unreadCount} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
            </Tooltip>

            <Tooltip title="Profile" arrow>
              <IconButton
                onClick={handleProfileDialogOpen}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                  },
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: 'rgba(255,255,255,0.2)',
                    border: '2px solid rgba(255, 255, 255, 0.5)',
                    fontWeight: 600,
                  }}
                >
                  {profileData.full_name?.charAt(0) || 'U'}
                </Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Notifications Menu */}
      <Menu
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={handleNotificationMenuClose}
        PaperProps={{
          elevation: 3,
          sx: {
            width: 350,
            maxHeight: 400,
            mt: 1.5,
            borderRadius: 2,
          },
        }}
      >
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h6" fontWeight={600}>
            Notifications
          </Typography>
        </Box>
        {notifications.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography color="text.secondary">No notifications</Typography>
          </Box>
        ) : (
          notifications.map((notification) => (
            <MenuItem
              key={notification.id}
              onClick={() => {
                markNotificationAsRead(notification.id);
                handleNotificationMenuClose();
              }}
              sx={{
                bgcolor: notification.is_read ? 'transparent' : 'action.hover',
                borderLeft: notification.is_read ? 'none' : '3px solid #0277bd',
                py: 1.5,
                px: 2,
                '&:hover': {
                  bgcolor: 'action.selected',
                },
              }}
            >
              <Box sx={{ width: '100%' }}>
                <Typography variant="subtitle2" fontWeight={notification.is_read ? 400 : 600}>
                  {notification.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  {notification.message}
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mt: 0.5, display: 'block' }}
                >
                  {new Date(notification.created_at).toLocaleDateString()}
                </Typography>
              </Box>
            </MenuItem>
          ))
        )}
      </Menu>

      {/* Profile Dialog */}
      <Dialog
        open={profileDialog}
        onClose={handleProfileDialogClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 2,
          },
        }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Person color="primary" />
            <Typography variant="h6" fontWeight={600}>
              Profile Settings
            </Typography>
          </Box>
        </DialogTitle>
        <Divider />
        <DialogContent sx={{ pt: 3 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
            <TextField
              label="Full Name"
              value={profileData.full_name}
              onChange={(e) => setProfileData({ ...profileData, full_name: e.target.value })}
              disabled={!editMode}
              fullWidth
              variant="outlined"
            />
            <TextField
              label="Email"
              value={profileData.email}
              onChange={(e) => setProfileData({ ...profileData, email: e.target.value })}
              disabled={!editMode}
              fullWidth
              variant="outlined"
              type="email"
            />
            <TextField
              label="Username"
              value={profileData.username}
              disabled
              fullWidth
              variant="outlined"
              helperText="Username cannot be changed"
            />
            <FormControlLabel
              control={<Switch checked={darkMode} onChange={toggleDarkMode} color="primary" />}
              label="Dark Mode"
            />
          </Box>
        </DialogContent>
        <Divider />
        <DialogActions sx={{ p: 2, gap: 1 }}>
          <Button onClick={handleProfileDialogClose} color="inherit">
            Cancel
          </Button>
          {editMode ? (
            <Button
              onClick={handleProfileUpdate}
              variant="contained"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <Save />}
            >
              Save Changes
            </Button>
          ) : (
            <Button onClick={() => setEditMode(true)} variant="outlined" startIcon={<Edit />}>
              Edit Profile
            </Button>
          )}
          <Button onClick={handleLogout} color="error" variant="outlined" startIcon={<Logout />}>
            Logout
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default Navbar;