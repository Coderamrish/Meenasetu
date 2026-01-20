import React from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Grid, 
  Card, 
  CardContent,
  Avatar,
  Chip,
  Stack,
  Divider,
  LinearProgress,
  Paper,
  alpha
} from '@mui/material';
import CodeIcon from '@mui/icons-material/Code';
import PsychologyIcon from '@mui/icons-material/Psychology';
import StorageIcon from '@mui/icons-material/Storage';
import CloudIcon from '@mui/icons-material/Cloud';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import WavesIcon from '@mui/icons-material/Waves';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import GroupsIcon from '@mui/icons-material/Groups';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import GitHubIcon from '@mui/icons-material/GitHub';
import EmailIcon from '@mui/icons-material/Email';
import VerifiedIcon from '@mui/icons-material/Verified';

const About = () => {
  const techStack = [
    { name: 'React.js', level: 95, color: '#61dafb' },
    { name: 'Node.js', level: 92, color: '#68a063' },
    { name: 'Python', level: 98, color: '#3776ab' },
    { name: 'TensorFlow', level: 90, color: '#ff6f00' },
    { name: 'PyTorch', level: 88, color: '#ee4c2c' },
    { name: 'MongoDB', level: 85, color: '#47a248' },
    { name: 'PostgreSQL', level: 87, color: '#336791' },
    { name: 'AWS', level: 82, color: '#ff9900' },
  ];

  const features = [
    {
      icon: <PsychologyIcon sx={{ fontSize: 40 }} />,
      title: 'Advanced AI Models',
      description: 'State-of-the-art deep learning models for fish classification and disease detection with 98.5% accuracy',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    },
    {
      icon: <WavesIcon sx={{ fontSize: 40 }} />,
      title: 'Computer Vision',
      description: 'Cutting-edge image recognition technology trained on thousands of aquatic species images',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    },
    {
      icon: <StorageIcon sx={{ fontSize: 40 }} />,
      title: 'Big Data Analytics',
      description: 'Scalable data processing pipeline handling millions of data points for comprehensive insights',
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    },
    {
      icon: <CloudIcon sx={{ fontSize: 40 }} />,
      title: 'Cloud Infrastructure',
      description: 'Robust AWS-powered architecture ensuring 99.9% uptime and lightning-fast performance',
      gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    },
    {
      icon: <AutoAwesomeIcon sx={{ fontSize: 40 }} />,
      title: 'Real-time Processing',
      description: 'Instant AI predictions and analytics with response times under 2 seconds',
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    },
    {
      icon: <RocketLaunchIcon sx={{ fontSize: 40 }} />,
      title: 'Continuous Innovation',
      description: 'Regular updates with new features, models, and improvements based on latest research',
      gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
    },
  ];

  const milestones = [
    { year: '2024', title: 'Platform Launch', description: 'MeenaSetu AI officially launched' },
    { year: '2024', title: '50,000+ Predictions', description: 'Crossed major milestone in AI predictions' },
    { year: '2024', title: '98.5% Accuracy', description: 'Achieved industry-leading accuracy' },
    { year: '2025', title: 'Global Expansion', description: 'Serving researchers worldwide' },
  ];

  const achievements = [
    { icon: <EmojiEventsIcon />, text: 'Best AI Innovation 2024', color: '#fbbf24' },
    { icon: <VerifiedIcon />, text: 'ISO 27001 Certified', color: '#10b981' },
    { icon: <GroupsIcon />, text: '5,200+ Active Users', color: '#3b82f6' },
    { icon: <LightbulbIcon />, text: '15+ Research Papers', color: '#8b5cf6' },
  ];

  return (
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh', pb: 8 }}>
      {/* Hero Section */}
      <Box
        sx={{
          position: 'relative',
          overflow: 'hidden',
          background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
          color: 'white',
          py: { xs: 8, md: 12 },
          mb: 8,
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at top right, rgba(255,255,255,0.1) 0%, transparent 60%)',
          },
        }}
      >
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Chip
              icon={<AutoAwesomeIcon />}
              label="About MeenaSetu AI"
              sx={{
                mb: 3,
                backgroundColor: alpha('#fff', 0.2),
                color: 'white',
                fontWeight: 600,
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255,255,255,0.3)',
              }}
            />
            <Typography
              variant="h2"
              component="h1"
              gutterBottom
              sx={{
                fontWeight: 800,
                fontSize: { xs: '2.5rem', md: '3.5rem' },
                mb: 3,
              }}
            >
              Revolutionizing Aquatic Research
            </Typography>
            <Typography
              variant="h5"
              sx={{
                opacity: 0.95,
                maxWidth: 800,
                mx: 'auto',
                fontWeight: 400,
                lineHeight: 1.6,
              }}
            >
              Powered by cutting-edge AI and built with passion for marine conservation, 
              MeenaSetu AI is your intelligent companion in aquatic research and analysis
            </Typography>
          </Box>

          {/* Achievements Bar */}
          <Grid container spacing={2} sx={{ mt: 4 }}>
            {achievements.map((achievement, index) => (
              <Grid item xs={6} md={3} key={index}>
                <Paper
                  sx={{
                    p: 2,
                    textAlign: 'center',
                    background: alpha('#fff', 0.1),
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255,255,255,0.2)',
                    color: 'white',
                  }}
                >
                  <Box sx={{ color: achievement.color, mb: 1 }}>
                    {achievement.icon}
                  </Box>
                  <Typography variant="body2" fontWeight={600}>
                    {achievement.text}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      <Container maxWidth="lg">
        {/* Mission & Vision */}
        <Grid container spacing={4} sx={{ mb: 8 }}>
          <Grid item xs={12} md={6}>
            <Card
              sx={{
                height: '100%',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Typography variant="h4" gutterBottom fontWeight={700}>
                  Our Mission
                </Typography>
                <Typography variant="body1" sx={{ lineHeight: 1.8 }}>
                  To democratize access to advanced AI technology for aquatic research, 
                  making world-class fish species identification and disease detection 
                  accessible to researchers, conservationists, and aquaculture professionals 
                  worldwide. We're committed to protecting marine biodiversity through 
                  innovative technology.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card
              sx={{
                height: '100%',
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                color: 'white',
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Typography variant="h4" gutterBottom fontWeight={700}>
                  Our Vision
                </Typography>
                <Typography variant="body1" sx={{ lineHeight: 1.8 }}>
                  To become the world's leading AI platform for aquatic intelligence, 
                  enabling groundbreaking discoveries in marine biology, sustainable 
                  aquaculture, and ocean conservation. We envision a future where AI-powered 
                  insights help preserve our oceans for generations to come.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Platform Features */}
        <Box sx={{ mb: 8 }}>
          <Typography
            variant="h3"
            textAlign="center"
            gutterBottom
            fontWeight={700}
            sx={{ mb: 2 }}
          >
            Platform Features
          </Typography>
          <Typography
            variant="h6"
            textAlign="center"
            color="text.secondary"
            sx={{ mb: 6, maxWidth: 700, mx: 'auto' }}
          >
            Built with cutting-edge technology to deliver exceptional performance and accuracy
          </Typography>

          <Grid container spacing={3}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      '& .feature-icon': {
                        transform: 'scale(1.1) rotate(5deg)',
                      },
                    },
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      height: 4,
                      background: feature.gradient,
                    },
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box
                      className="feature-icon"
                      sx={{
                        width: 64,
                        height: 64,
                        borderRadius: 3,
                        background: feature.gradient,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: 2,
                        color: 'white',
                        transition: 'transform 0.3s ease',
                      }}
                    >
                      {feature.icon}
                    </Box>
                    <Typography variant="h6" gutterBottom fontWeight={700}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Creator Section */}
        <Card
          sx={{
            mb: 8,
            background: 'linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)',
            color: 'white',
            overflow: 'hidden',
            position: 'relative',
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              top: -50,
              right: -50,
              width: 200,
              height: 200,
              borderRadius: '50%',
              background: alpha('#fff', 0.05),
            }}
          />
          <CardContent sx={{ p: { xs: 4, md: 6 }, position: 'relative', zIndex: 1 }}>
            <Grid container spacing={4} alignItems="center">
              <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
                <Avatar
                  sx={{
                    width: 180,
                    height: 180,
                    mx: 'auto',
                    mb: 2,
                    border: '4px solid rgba(255,255,255,0.2)',
                    background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                    fontSize: '4rem',
                    fontWeight: 800,
                  }}
                >
                  AT
                </Avatar>
                <Stack direction="row" spacing={1} justifyContent="center">
                  <Avatar sx={{ bgcolor: '#0077b5', width: 36, height: 36, cursor: 'pointer' }}>
                    <LinkedInIcon fontSize="small" />
                  </Avatar>
                  <Avatar sx={{ bgcolor: '#333', width: 36, height: 36, cursor: 'pointer' }}>
                    <GitHubIcon fontSize="small" />
                  </Avatar>
                  <Avatar sx={{ bgcolor: '#ea4335', width: 36, height: 36, cursor: 'pointer' }}>
                    <EmailIcon fontSize="small" />
                  </Avatar>
                </Stack>
              </Grid>

              <Grid item xs={12} md={8}>
                <Chip
                  label="Creator & Lead Developer"
                  size="small"
                  sx={{
                    mb: 2,
                    backgroundColor: alpha('#fff', 0.2),
                    color: 'white',
                    fontWeight: 600,
                  }}
                />
                <Typography variant="h3" gutterBottom fontWeight={800}>
                  Amrish Kumar Tiwary
                </Typography>
                <Typography variant="h6" gutterBottom sx={{ opacity: 0.9, mb: 3 }}>
                  Full Stack AI Engineer | Machine Learning Expert
                </Typography>
                <Typography variant="body1" paragraph sx={{ lineHeight: 1.8, opacity: 0.95 }}>
                  A passionate Full Stack AI Engineer with expertise in building intelligent systems 
                  that solve real-world problems. With a deep understanding of machine learning, 
                  computer vision, and full-stack development, Amrish created MeenaSetu AI to bridge 
                  the gap between cutting-edge AI technology and aquatic research.
                </Typography>
                <Typography variant="body1" sx={{ lineHeight: 1.8, opacity: 0.95 }}>
                  Specializing in Python, TensorFlow, PyTorch, React, and cloud technologies, 
                  he's dedicated to leveraging AI for environmental conservation and sustainable 
                  development. MeenaSetu AI represents his vision of making advanced technology 
                  accessible to researchers and conservationists worldwide.
                </Typography>

                <Stack direction="row" spacing={2} sx={{ mt: 3, flexWrap: 'wrap', gap: 1 }}>
                  {['AI/ML', 'Deep Learning', 'Full Stack', 'Cloud Computing', 'Computer Vision'].map((skill, index) => (
                    <Chip
                      key={index}
                      label={skill}
                      sx={{
                        backgroundColor: alpha('#fff', 0.2),
                        color: 'white',
                        fontWeight: 600,
                        border: '1px solid rgba(255,255,255,0.3)',
                      }}
                    />
                  ))}
                </Stack>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Tech Stack */}
        <Box sx={{ mb: 8 }}>
          <Typography variant="h4" textAlign="center" gutterBottom fontWeight={700} sx={{ mb: 2 }}>
            Technology Stack
          </Typography>
          <Typography
            variant="body1"
            textAlign="center"
            color="text.secondary"
            sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}
          >
            Built with industry-leading technologies and frameworks
          </Typography>

          <Grid container spacing={3}>
            {techStack.map((tech, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body1" fontWeight={600}>
                        {tech.name}
                      </Typography>
                      <Typography variant="body2" fontWeight={700} color={tech.color}>
                        {tech.level}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={tech.level}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: alpha(tech.color, 0.1),
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: tech.color,
                          borderRadius: 4,
                        },
                      }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Timeline */}
        <Box sx={{ mb: 8 }}>
          <Typography variant="h4" textAlign="center" gutterBottom fontWeight={700} sx={{ mb: 6 }}>
            Our Journey
          </Typography>

          <Grid container spacing={3}>
            {milestones.map((milestone, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    position: 'relative',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: 4,
                      height: '100%',
                      background: 'linear-gradient(180deg, #0277bd 0%, #00897b 100%)',
                    },
                  }}
                >
                  <CardContent sx={{ pl: 3 }}>
                    <Chip
                      label={milestone.year}
                      size="small"
                      sx={{
                        mb: 2,
                        background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
                        color: 'white',
                        fontWeight: 700,
                      }}
                    />
                    <Typography variant="h6" gutterBottom fontWeight={700}>
                      {milestone.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {milestone.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* CTA Section */}
        <Card
          sx={{
            p: 6,
            textAlign: 'center',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
          }}
        >
          <Typography variant="h4" gutterBottom fontWeight={700}>
            Join the AI Revolution in Aquatic Research
          </Typography>
          <Typography variant="h6" paragraph sx={{ opacity: 0.95, mb: 4 }}>
            Be part of a global community using AI to protect our oceans
          </Typography>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center">
            <Chip
              icon={<GroupsIcon />}
              label="5,200+ Active Researchers"
              sx={{
                backgroundColor: alpha('#fff', 0.2),
                color: 'white',
                fontWeight: 600,
                fontSize: '1rem',
                py: 2.5,
              }}
            />
            <Chip
              icon={<EmojiEventsIcon />}
              label="50,000+ Predictions Made"
              sx={{
                backgroundColor: alpha('#fff', 0.2),
                color: 'white',
                fontWeight: 600,
                fontSize: '1rem',
                py: 2.5,
              }}
            />
          </Stack>
        </Card>
      </Container>
    </Box>
  );
};

export default About;