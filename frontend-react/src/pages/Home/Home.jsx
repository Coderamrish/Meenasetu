import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Paper,
  alpha,
  Divider,
} from '@mui/material';
import {
  Pets,
  Healing,
  Chat,
  BarChart,
  UploadFile,
  Analytics,
  CheckCircle,
  Storage,
  SmartToy,
  Api,
  ArrowForward,
  Circle,
  TrendingUp,
  Speed,
  Security,
  CloudDone,
  AutoAwesome,
  EmojiEvents,
  Lightbulb,
} from '@mui/icons-material';

const Home = () => {
  const navigate = useNavigate();

  const systemStatusCards = [
    { 
      title: 'AI Core', 
      status: 'Operational', 
      icon: <SmartToy />, 
      color: '#4caf50',
      metric: '99.9%',
      label: 'Uptime'
    },
    { 
      title: 'Vector DB', 
      status: 'Connected', 
      icon: <Storage />, 
      color: '#2196f3',
      metric: '1.2M',
      label: 'Documents'
    },
    { 
      title: 'ML Models', 
      status: 'Ready', 
      icon: <Pets />, 
      color: '#ff9800',
      metric: '31',
      label: 'Species'
    },
    { 
      title: 'API', 
      status: 'v1.0.0', 
      icon: <Api />, 
      color: '#9c27b0',
      metric: '<100ms',
      label: 'Response'
    },
  ];

  const featureCards = [
    {
      icon: <Pets sx={{ fontSize: 48 }} />,
      title: 'Fish Species Classification',
      description: 'Identify 31 different fish species using advanced EfficientNet CNN models with 94% accuracy.',
      route: '/data',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    },
    {
      icon: <Healing sx={{ fontSize: 48 }} />,
      title: 'Disease Detection',
      description: 'Detect fish diseases early and receive AI-powered treatment recommendations instantly.',
      route: '/data',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    },
    {
      icon: <Chat sx={{ fontSize: 48 }} />,
      title: 'Conversational AI',
      description: 'Ask questions and get intelligent answers with our RAG-based conversational system.',
      route: '/chatbot',
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    },
    {
      icon: <BarChart sx={{ fontSize: 48 }} />,
      title: 'Smart Visualizations',
      description: 'Generate dynamic charts and visual analytics from your aquaculture data automatically.',
      route: '/analytics',
      gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    },
    {
      icon: <UploadFile sx={{ fontSize: 48 }} />,
      title: 'Document Intelligence',
      description: 'Upload PDFs, CSVs, and images for AI-powered data extraction and analysis.',
      route: '/data',
      gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    },
    {
      icon: <Analytics sx={{ fontSize: 48 }} />,
      title: 'Data Analytics',
      description: 'Comprehensive analytics dashboard for monitoring farm performance and trends.',
      route: '/analytics',
      gradient: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
    },
  ];

  const capabilities = [
    { text: 'RAG-based Q&A with context-aware responses', icon: <Chat /> },
    { text: 'CNN-based disease detection with treatment suggestions', icon: <Healing /> },
    { text: 'EfficientNet image classification for 31 fish species', icon: <Pets /> },
    { text: 'ChromaDB vector search for semantic document retrieval', icon: <Storage /> },
    { text: 'Conversation memory for multi-turn dialogues', icon: <SmartToy /> },
    { text: 'Automatic visualization generation from data', icon: <BarChart /> },
  ];

  const stats = [
    { value: '31', label: 'Fish Species', icon: <Pets />, color: '#667eea' },
    { value: '94%', label: 'Accuracy', icon: <EmojiEvents />, color: '#f5576c' },
    { value: '1.2M', label: 'Documents', icon: <Storage />, color: '#00f2fe' },
    { value: '<100ms', label: 'Response Time', icon: <Speed />, color: '#38f9d7' },
  ];

  const whyChooseUs = [
    {
      icon: <Security />,
      title: 'Enterprise Security',
      description: 'Bank-grade encryption and secure data handling',
    },
    {
      icon: <Speed />,
      title: 'Lightning Fast',
      description: 'Real-time predictions under 100ms response time',
    },
    {
      icon: <CloudDone />,
      title: 'Cloud Native',
      description: '99.9% uptime with scalable infrastructure',
    },
    {
      icon: <AutoAwesome />,
      title: 'AI-Powered',
      description: 'State-of-the-art machine learning models',
    },
  ];

  return (
    <Box sx={{ bgcolor: '#f5f7fa', minHeight: '100vh' }}>
      {/* Hero Section with Gradient Background */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
          color: 'white',
          pt: 8,
          pb: 12,
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%)',
          },
          '&::after': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 0%, transparent 50%)',
          },
        }}
      >
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Chip
              label="AI-Powered Aquaculture Platform"
              sx={{
                mb: 3,
                bgcolor: 'rgba(255,255,255,0.2)',
                color: 'white',
                fontWeight: 600,
                fontSize: '0.9rem',
                backdropFilter: 'blur(10px)',
              }}
            />
            <Typography
              variant="h1"
              sx={{
                fontWeight: 800,
                mb: 3,
                fontSize: { xs: '2.5rem', md: '4rem' },
                letterSpacing: '-0.02em',
                textShadow: '0 4px 20px rgba(0,0,0,0.2)',
              }}
            >
              MeenaSetu
            </Typography>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 300,
                mb: 2,
                fontSize: { xs: '1.2rem', md: '2rem' },
                opacity: 0.95,
              }}
            >
              Intelligent Aquatic Expert
            </Typography>
            <Typography
              variant="h6"
              sx={{
                mb: 5,
                fontWeight: 400,
                fontSize: { xs: '1rem', md: '1.3rem' },
                opacity: 0.9,
                maxWidth: '800px',
                mx: 'auto',
              }}
            >
              Transform your aquaculture operations with AI-powered insights, disease detection, and smart analytics
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => navigate('/chatbot')}
                sx={{
                  bgcolor: 'white',
                  color: '#0277bd',
                  px: 5,
                  py: 2,
                  fontSize: '1.1rem',
                  fontWeight: 600,
                  '&:hover': { 
                    bgcolor: '#f5f5f5',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                  },
                  transition: 'all 0.3s',
                  borderRadius: 2,
                }}
                endIcon={<Chat />}
              >
                Start Conversation
              </Button>
              <Button
                variant="outlined"
                size="large"
                onClick={() => navigate('/data')}
                sx={{
                  borderColor: 'white',
                  color: 'white',
                  px: 5,
                  py: 2,
                  fontSize: '1.1rem',
                  fontWeight: 600,
                  borderWidth: 2,
                  '&:hover': { 
                    borderColor: 'white',
                    bgcolor: 'rgba(255,255,255,0.1)',
                    borderWidth: 2,
                    transform: 'translateY(-2px)',
                  },
                  transition: 'all 0.3s',
                  borderRadius: 2,
                }}
                endIcon={<UploadFile />}
              >
                Upload Data
              </Button>
            </Box>
          </Box>

          {/* Stats Row */}
          <Grid container spacing={3} sx={{ mt: 4 }}>
            {stats.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    textAlign: 'center',
                    background: 'rgba(255,255,255,0.15)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: 3,
                    transition: 'all 0.3s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      background: 'rgba(255,255,255,0.25)',
                    },
                  }}
                >
                  <Box sx={{ color: stat.color, mb: 1 }}>
                    {React.cloneElement(stat.icon, { sx: { fontSize: 32 } })}
                  </Box>
                  <Typography variant="h3" sx={{ fontWeight: 700, mb: 0.5 }}>
                    {stat.value}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    {stat.label}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ mt: -4, position: 'relative', zIndex: 2 }}>
        {/* System Status Cards */}
        <Box sx={{ mb: 8 }}>
          <Grid container spacing={3}>
            {systemStatusCards.map((card, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    boxShadow: 3,
                    borderRadius: 3,
                    transition: 'all 0.3s',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: 6,
                    },
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Box
                      sx={{
                        bgcolor: alpha(card.color, 0.1),
                        borderRadius: '50%',
                        width: 60,
                        height: 60,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mx: 'auto',
                        mb: 2,
                        color: card.color,
                      }}
                    >
                      {React.cloneElement(card.icon, { sx: { fontSize: 32 } })}
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 1, color: '#37474f' }}>
                      {card.title}
                    </Typography>
                    <Chip
                      label={card.status}
                      size="small"
                      sx={{
                        bgcolor: card.color,
                        color: 'white',
                        fontWeight: 600,
                        mb: 2,
                      }}
                    />
                    <Divider sx={{ my: 1.5 }} />
                    <Typography variant="h5" sx={{ fontWeight: 700, color: card.color }}>
                      {card.metric}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {card.label}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Feature Cards Grid */}
        <Box sx={{ mb: 8 }}>
          <Box sx={{ textAlign: 'center', mb: 5 }}>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                color: '#1a237e',
                mb: 2,
                fontSize: { xs: '2rem', md: '2.5rem' },
              }}
            >
              Platform Features
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Comprehensive AI-powered tools designed for modern aquaculture management
            </Typography>
          </Box>
          <Grid container spacing={3}>
            {featureCards.map((feature, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    boxShadow: 2,
                    borderRadius: 3,
                    transition: 'all 0.3s',
                    position: 'relative',
                    overflow: 'hidden',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: 6,
                      '& .feature-overlay': {
                        opacity: 1,
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
                  <Box
                    className="feature-overlay"
                    sx={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      background: `${feature.gradient}`,
                      opacity: 0,
                      transition: 'opacity 0.3s',
                      zIndex: 0,
                    }}
                  />
                  <CardContent
                    sx={{
                      flexGrow: 1,
                      textAlign: 'center',
                      p: 4,
                      position: 'relative',
                      zIndex: 1,
                    }}
                  >
                    <Box
                      sx={{
                        background: feature.gradient,
                        borderRadius: '20px',
                        width: 80,
                        height: 80,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mx: 'auto',
                        mb: 3,
                        color: 'white',
                        boxShadow: `0 8px 24px ${alpha('#0277bd', 0.3)}`,
                      }}
                    >
                      {feature.icon}
                    </Box>
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 700,
                        mb: 2,
                        color: '#1a237e',
                      }}
                    >
                      {feature.title}
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{
                        color: '#546e7a',
                        mb: 3,
                        lineHeight: 1.7,
                      }}
                    >
                      {feature.description}
                    </Typography>
                    <Button
                      variant="contained"
                      size="medium"
                      onClick={() => navigate(feature.route)}
                      endIcon={<ArrowForward />}
                      sx={{
                        mt: 'auto',
                        background: feature.gradient,
                        fontWeight: 600,
                        '&:hover': {
                          opacity: 0.9,
                        },
                      }}
                    >
                      Explore
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Why Choose Us Section */}
        <Box sx={{ mb: 8 }}>
          <Box sx={{ textAlign: 'center', mb: 5 }}>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                color: '#1a237e',
                mb: 2,
                fontSize: { xs: '2rem', md: '2.5rem' },
              }}
            >
              Why Choose MeenaSetu?
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Built with cutting-edge technology for reliability and performance
            </Typography>
          </Box>
          <Grid container spacing={4}>
            {whyChooseUs.map((item, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Box sx={{ textAlign: 'center' }}>
                  <Box
                    sx={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      borderRadius: '50%',
                      width: 80,
                      height: 80,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mx: 'auto',
                      mb: 2,
                      color: 'white',
                      boxShadow: '0 8px 24px rgba(102, 126, 234, 0.4)',
                    }}
                  >
                    {React.cloneElement(item.icon, { sx: { fontSize: 40 } })}
                  </Box>
                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: '#1a237e' }}>
                    {item.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {item.description}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Two Column Layout: Capabilities & Quick Actions */}
        <Grid container spacing={4} sx={{ mb: 8 }}>
          {/* Platform Capabilities */}
          <Grid item xs={12} md={7}>
            <Card sx={{ boxShadow: 3, borderRadius: 3, height: '100%' }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Lightbulb sx={{ fontSize: 32, color: '#ff9800', mr: 2 }} />
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a237e' }}>
                    Platform Capabilities
                  </Typography>
                </Box>
                <List>
                  {capabilities.map((capability, index) => (
                    <ListItem
                      key={index}
                      sx={{
                        py: 1.5,
                        px: 2,
                        mb: 1,
                        borderRadius: 2,
                        transition: 'all 0.2s',
                        '&:hover': {
                          bgcolor: alpha('#0277bd', 0.05),
                          transform: 'translateX(8px)',
                        },
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        <Box
                          sx={{
                            bgcolor: alpha('#0277bd', 0.1),
                            borderRadius: 1.5,
                            p: 0.5,
                            display: 'flex',
                            color: '#0277bd',
                          }}
                        >
                          {React.cloneElement(capability.icon, { sx: { fontSize: 20 } })}
                        </Box>
                      </ListItemIcon>
                      <ListItemText
                        primary={capability.text}
                        primaryTypographyProps={{
                          variant: 'body1',
                          fontWeight: 500,
                          color: '#37474f',
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Quick Actions */}
          <Grid item xs={12} md={5}>
            <Card
              sx={{
                boxShadow: 3,
                borderRadius: 3,
                height: '100%',
                background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <AutoAwesome sx={{ fontSize: 32, color: '#9c27b0', mr: 2 }} />
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a237e' }}>
                    Quick Actions
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => navigate('/chatbot')}
                    startIcon={<Chat />}
                    sx={{
                      justifyContent: 'flex-start',
                      py: 2,
                      px: 3,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      fontWeight: 600,
                      '&:hover': {
                        transform: 'translateX(4px)',
                      },
                      transition: 'all 0.2s',
                    }}
                  >
                    Open AI Chatbot
                  </Button>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => navigate('/analytics')}
                    startIcon={<Analytics />}
                    sx={{
                      justifyContent: 'flex-start',
                      py: 2,
                      px: 3,
                      background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                      fontWeight: 600,
                      '&:hover': {
                        transform: 'translateX(4px)',
                      },
                      transition: 'all 0.2s',
                    }}
                  >
                    View Analytics Dashboard
                  </Button>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => navigate('/data')}
                    startIcon={<UploadFile />}
                    sx={{
                      justifyContent: 'flex-start',
                      py: 2,
                      px: 3,
                      background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                      fontWeight: 600,
                      '&:hover': {
                        transform: 'translateX(4px)',
                      },
                      transition: 'all 0.2s',
                    }}
                  >
                    Manage Data & Files
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => navigate('/profile')}
                    startIcon={<CheckCircle />}
                    sx={{
                      justifyContent: 'flex-start',
                      py: 2,
                      px: 3,
                      borderWidth: 2,
                      borderColor: '#9c27b0',
                      color: '#9c27b0',
                      fontWeight: 600,
                      '&:hover': {
                        borderWidth: 2,
                        transform: 'translateX(4px)',
                        borderColor: '#7b1fa2',
                      },
                      transition: 'all 0.2s',
                    }}
                  >
                    My Profile Settings
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Final CTA Section */}
        <Box
          sx={{
            textAlign: 'center',
            background: 'linear-gradient(135deg, #0277bd 0%, #00897b 100%)',
            color: 'white',
            p: { xs: 4, md: 8 },
            borderRadius: 4,
            boxShadow: '0 12px 40px rgba(2, 119, 189, 0.3)',
            position: 'relative',
            overflow: 'hidden',
            mb: 6,
            '&::before': {
              content: '""',
              position: 'absolute',
              top: -100,
              right: -100,
              width: 300,
              height: 300,
              background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
            },
          }}
        >
          <Box sx={{ position: 'relative', zIndex: 1 }}>
            <EmojiEvents sx={{ fontSize: 64, mb: 2, opacity: 0.9 }} />
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                mb: 2,
                fontSize: { xs: '1.8rem', md: '2.5rem' },
              }}
            >
              Ready to Transform Your Aquaculture?
            </Typography>
            <Typography
              variant="h6"
              sx={{
                mb: 4,
                fontSize: { xs: '1rem', md: '1.2rem' },
                opacity: 0.95,
                maxWidth: 700,
                mx: 'auto',
                fontWeight: 300,
              }}
            >
              Join farmers and researchers worldwide in leveraging AI for smarter, more sustainable fisheries. Start making data-driven decisions today with MeenaSetu.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => navigate('/chatbot')}
                sx={{
                  bgcolor: 'white',
                  color: '#0277bd',
                  px: 6,
                  py: 2,
                  fontSize: '1.2rem',
                  fontWeight: 700,
                  borderRadius: 2,
                  '&:hover': {
                    bgcolor: '#f5f5f5',
                    transform: 'translateY(-4px)',
                    boxShadow: '0 12px 32px rgba(0,0,0,0.3)',
                  },
                  transition: 'all 0.3s',
                }}
                endIcon={<ArrowForward />}
              >
                Start Using MeenaSetu AI
              </Button>
              <Button
                variant="outlined"
                size="large"
                onClick={() => navigate('/about')}
                sx={{
                  borderColor: 'white',
                  color: 'white',
                  px: 6,
                  py: 2,
                  fontSize: '1.2rem',
                  fontWeight: 700,
                  borderWidth: 2,
                  borderRadius: 2,
                  '&:hover': {
                    borderColor: 'white',
                    bgcolor: 'rgba(255,255,255,0.1)',
                    borderWidth: 2,
                    transform: 'translateY(-4px)',
                  },
                  transition: 'all 0.3s',
                }}
              >
                Learn More
              </Button>
            </Box>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Home;