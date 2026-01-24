import React, { useState, useEffect } from 'react';
import { 
  User, Camera, Mail, Phone, MapPin, Briefcase, Trophy, 
  TrendingUp, Activity, Database, Fish, Microscope, Settings,
  Download, RefreshCw, Edit, Save, X, CheckCircle, Lock,
  Clock, Calendar, BarChart2, Award, Target, Zap, Star,
  MessageSquare, FileText, Image, AlertCircle, Bell, Shield,
  Eye, ChevronRight, Upload, Share2, Globe, Heart
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const EnhancedProfile = () => {
  const [editMode, setEditMode] = useState(false);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [showAvatarPicker, setShowAvatarPicker] = useState(false);
  const [notification, setNotification] = useState({ show: false, message: '', type: 'success' });

  // API Data States
  const [apiStats, setApiStats] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
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
    avatar: '🐠'
  });

  const [stats, setStats] = useState({
    queriesAsked: 0,
    fishIdentified: 0,
    diseasesDetected: 0,
    documentsUploaded: 0,
    expertiseLevel: 0,
  });

  const GRADIENTS = {
    purple: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    green: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    orange: 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)',
    blue: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    pink: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    gold: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
  };

  const avatarOptions = ['🐠', '🐟', '🐡', '🦈', '🐙', '🦀', '🦞', '🐚', '🐋', '🦑', '🐬', '🦭'];

  const achievements = [
    { title: 'First Query', icon: '🎯', desc: 'Asked your first question', threshold: 1, current: 0, field: 'queriesAsked', gradient: GRADIENTS.blue },
    { title: 'Fish Expert', icon: '🐟', desc: '50+ species identified', threshold: 50, current: 31, field: 'fishIdentified', gradient: GRADIENTS.green },
    { title: 'Disease Detective', icon: '🔬', desc: '25+ diseases detected', threshold: 25, current: 6, field: 'diseasesDetected', gradient: GRADIENTS.orange },
    { title: 'Knowledge Master', icon: '📚', desc: 'Upload 50 documents', threshold: 50, current: 0, field: 'documentsUploaded', gradient: GRADIENTS.pink },
    { title: 'AI Whisperer', icon: '🤖', desc: 'Complete 500 queries', threshold: 500, current: 0, field: 'queriesAsked', gradient: GRADIENTS.purple },
    { title: 'Research Pioneer', icon: '🏆', desc: 'Master level achieved', threshold: 100, current: 0, field: 'expertiseLevel', gradient: GRADIENTS.gold },
  ];

  useEffect(() => {
    fetchAPIData();
  }, []);

  const fetchAPIData = async () => {
    setLoading(true);
    try {
      const [statsRes, historyRes, speciesRes, diseaseRes] = await Promise.all([
        fetch(`${API_BASE}/stats`).catch(() => null),
        fetch(`${API_BASE}/conversation/history?limit=50`).catch(() => null),
        fetch(`${API_BASE}/docs/species-list`).catch(() => null),
        fetch(`${API_BASE}/docs/diseases`).catch(() => null)
      ]);

      const [statsData, historyData, speciesData, diseaseData] = await Promise.all([
        statsRes?.ok ? statsRes.json() : null,
        historyRes?.ok ? historyRes.json() : null,
        speciesRes?.ok ? speciesRes.json() : null,
        diseaseRes?.ok ? diseaseRes.json() : null
      ]);

      if (statsData) {
        setApiStats(statsData);
        const queries = statsData.statistics?.session_info?.queries_processed || 0;
        const docs = statsData.statistics?.database_stats?.total_documents || 0;
        
        setStats(prev => ({
          ...prev,
          queriesAsked: queries,
          documentsUploaded: docs,
          expertiseLevel: Math.min(Math.floor((queries / 10) * 100), 100),
        }));
      }

      if (historyData?.history) setConversationHistory(historyData.history);
      if (speciesData?.species) setSpeciesList(speciesData.species);
      if (diseaseData?.detectable_diseases) setDiseasesList(diseaseData.detectable_diseases);

      showNotification('Profile data loaded successfully!', 'success');
    } catch (error) {
      console.error('Error fetching data:', error);
      showNotification('Failed to load data', 'error');
    } finally {
      setLoading(false);
    }
  };

  const showNotification = (message, type) => {
    setNotification({ show: true, message, type });
    setTimeout(() => setNotification({ show: false, message: '', type: 'success' }), 3000);
  };

  const handleSave = () => {
    setEditMode(false);
    showNotification('Profile updated successfully!', 'success');
  };

  const formatTime = (timestamp) => {
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

  const StatCard = ({ icon: Icon, label, value, trend, gradient, delay }) => (
    <div style={{
      background: gradient,
      borderRadius: '20px',
      padding: '2rem',
      color: 'white',
      position: 'relative',
      overflow: 'hidden',
      cursor: 'pointer',
      transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      animation: `slideUp 0.6s ease ${delay}s both`
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = 'translateY(-8px) scale(1.02)';
      e.currentTarget.style.boxShadow = '0 20px 60px rgba(0,0,0,0.25)';
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = 'translateY(0) scale(1)';
      e.currentTarget.style.boxShadow = '0 10px 40px rgba(0,0,0,0.15)';
    }}>
      <div style={{
        position: 'absolute',
        top: -60,
        right: -60,
        width: 150,
        height: 150,
        borderRadius: '50%',
        background: 'rgba(255, 255, 255, 0.1)',
        filter: 'blur(20px)'
      }} />
      
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem', opacity: 0.9 }}>
          <Icon size={40} />
        </div>
        <div style={{ fontSize: '3rem', fontWeight: '800', marginBottom: '0.5rem', lineHeight: 1 }}>
          {value}
        </div>
        <div style={{ fontSize: '0.9rem', opacity: 0.95, fontWeight: '600', letterSpacing: '0.5px' }}>
          {label}
        </div>
        {trend && (
          <div style={{
            marginTop: '0.75rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 0.8rem',
            background: 'rgba(255,255,255,0.2)',
            borderRadius: '20px',
            fontSize: '0.85rem',
            fontWeight: 'bold'
          }}>
            <TrendingUp size={14} />
            +{trend}%
          </div>
        )}
      </div>
    </div>
  );

  const AchievementCard = ({ achievement, index }) => {
    const progress = Math.min((achievement.current / achievement.threshold) * 100, 100);
    const unlocked = achievement.current >= achievement.threshold;

    return (
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '2rem',
        boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
        position: 'relative',
        overflow: 'hidden',
        opacity: unlocked ? 1 : 0.7,
        border: unlocked ? '3px solid #667eea' : '2px solid #e2e8f0',
        transition: 'all 0.3s ease',
        cursor: 'pointer',
        animation: `fadeIn 0.6s ease ${index * 0.1}s both`
      }}
      onMouseEnter={(e) => {
        if (unlocked) {
          e.currentTarget.style.transform = 'translateY(-8px)';
          e.currentTarget.style.boxShadow = '0 16px 48px rgba(0,0,0,0.15)';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.08)';
      }}>
        {unlocked && (
          <div style={{
            position: 'absolute',
            top: 10,
            right: 10,
            background: '#10b981',
            borderRadius: '50%',
            width: 32,
            height: 32,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            boxShadow: '0 4px 12px rgba(16, 185, 129, 0.4)'
          }}>
            <CheckCircle size={20} />
          </div>
        )}
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>
            {unlocked ? achievement.icon : '🔒'}
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: '700', marginBottom: '0.5rem', color: '#1e293b' }}>
            {achievement.title}
          </div>
          <div style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '1rem' }}>
            {achievement.desc}
          </div>
          
          {!unlocked && (
            <>
              <div style={{
                width: '100%',
                height: 8,
                background: '#e2e8f0',
                borderRadius: 10,
                overflow: 'hidden',
                marginBottom: '0.5rem'
              }}>
                <div style={{
                  width: `${progress}%`,
                  height: '100%',
                  background: achievement.gradient,
                  borderRadius: 10,
                  transition: 'width 0.6s ease'
                }} />
              </div>
              <div style={{ fontSize: '0.85rem', color: '#64748b', fontWeight: '600' }}>
                {achievement.current} / {achievement.threshold} ({Math.round(progress)}%)
              </div>
            </>
          )}
          
          {unlocked && (
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              background: '#10b98120',
              color: '#10b981',
              borderRadius: '20px',
              fontSize: '0.85rem',
              fontWeight: '700'
            }}>
              <CheckCircle size={16} />
              Unlocked!
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom, #f8fafc 0%, #e2e8f0 100%)',
      padding: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Notification */}
      {notification.show && (
        <div style={{
          position: 'fixed',
          top: 20,
          right: 20,
          background: notification.type === 'success' ? '#10b981' : '#ef4444',
          color: 'white',
          padding: '1rem 1.5rem',
          borderRadius: '12px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
          zIndex: 1000,
          animation: 'slideInRight 0.3s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontWeight: '600'
        }}>
          {notification.type === 'success' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
          {notification.message}
        </div>
      )}

      {/* Header Cover */}
      <div style={{
        background: GRADIENTS.purple,
        borderRadius: '24px',
        padding: '3rem 2rem',
        marginBottom: '2rem',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)'
      }}>
        <div style={{
          position: 'absolute',
          top: -100,
          right: -100,
          width: 300,
          height: 300,
          borderRadius: '50%',
          background: 'rgba(255,255,255,0.1)',
          filter: 'blur(40px)'
        }} />
        <div style={{
          position: 'absolute',
          bottom: -80,
          left: -80,
          width: 250,
          height: 250,
          borderRadius: '50%',
          background: 'rgba(255,255,255,0.1)',
          filter: 'blur(40px)'
        }} />

        <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
            {/* Avatar */}
            <div style={{ position: 'relative' }}>
              <div style={{
                width: 140,
                height: 140,
                borderRadius: '50%',
                background: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '5rem',
                boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
                border: '5px solid white',
                cursor: 'pointer'
              }}
              onClick={() => setShowAvatarPicker(true)}>
                {userData.avatar}
              </div>
              <button style={{
                position: 'absolute',
                bottom: 5,
                right: 5,
                width: 40,
                height: 40,
                borderRadius: '50%',
                background: 'white',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                transition: 'all 0.3s ease'
              }}
              onClick={() => setShowAvatarPicker(true)}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}>
                <Camera size={20} color="#667eea" />
              </button>
            </div>

            {/* User Info */}
            <div style={{ color: 'white' }}>
              {editMode ? (
                <input
                  value={userData.name}
                  onChange={(e) => setUserData({...userData, name: e.target.value})}
                  style={{
                    background: 'white',
                    border: 'none',
                    borderRadius: '12px',
                    padding: '0.75rem 1rem',
                    fontSize: '2.5rem',
                    fontWeight: '800',
                    marginBottom: '0.5rem',
                    width: '100%'
                  }}
                />
              ) : (
                <h1 style={{ margin: 0, fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                  {userData.name}
                </h1>
              )}
              
              {editMode ? (
                <input
                  value={userData.title}
                  onChange={(e) => setUserData({...userData, title: e.target.value})}
                  style={{
                    background: 'white',
                    border: 'none',
                    borderRadius: '12px',
                    padding: '0.5rem 1rem',
                    fontSize: '1.1rem',
                    width: '100%'
                  }}
                />
              ) : (
                <p style={{ margin: 0, fontSize: '1.1rem', opacity: 0.95, fontWeight: '500' }}>
                  {userData.title}
                </p>
              )}

              <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap' }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  background: 'rgba(255,255,255,0.25)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  fontSize: '0.9rem',
                  fontWeight: '600'
                }}>
                  <MapPin size={16} />
                  {userData.location}
                </div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  background: 'rgba(255,255,255,0.25)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  fontSize: '0.9rem',
                  fontWeight: '600'
                }}>
                  <Briefcase size={16} />
                  {userData.organization}
                </div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  background: 'rgba(255,255,255,0.25)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  fontSize: '0.9rem',
                  fontWeight: '600'
                }}>
                  <Calendar size={16} />
                  Member since {userData.joinDate}
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '1rem' }}>
            {editMode ? (
              <>
                <button onClick={handleSave} style={{
                  padding: '0.75rem 1.5rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: 'white',
                  color: '#667eea',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
                  <Save size={18} />
                  Save Changes
                </button>
                <button onClick={() => setEditMode(false)} style={{
                  padding: '0.75rem 1.5rem',
                  borderRadius: '12px',
                  border: '2px solid white',
                  background: 'transparent',
                  color: 'white',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                  <X size={18} />
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button onClick={() => setEditMode(true)} style={{
                  padding: '0.75rem 1.5rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: 'white',
                  color: '#667eea',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
                  <Edit size={18} />
                  Edit Profile
                </button>
                <button onClick={fetchAPIData} disabled={loading} style={{
                  padding: '0.75rem 1.5rem',
                  borderRadius: '12px',
                  border: '2px solid white',
                  background: loading ? 'rgba(255,255,255,0.15)' : 'transparent',
                  color: 'white',
                  fontWeight: '700',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => !loading && (e.currentTarget.style.background = 'rgba(255,255,255,0.15)')}
                onMouseLeave={(e) => !loading && (e.currentTarget.style.background = 'transparent')}>
                  <RefreshCw size={18} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
                  {loading ? 'Loading...' : 'Refresh'}
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '1.5rem',
        marginBottom: '2rem'
      }}>
        <StatCard icon={MessageSquare} label="Total Queries" value={stats.queriesAsked} trend={12} gradient={GRADIENTS.purple} delay={0} />
        <StatCard icon={Fish} label="Species Known" value={speciesList.length || 31} trend={8} gradient={GRADIENTS.green} delay={0.1} />
        <StatCard icon={Microscope} label="Diseases Tracked" value={diseasesList.length || 6} gradient={GRADIENTS.orange} delay={0.2} />
        <StatCard icon={FileText} label="Documents" value={stats.documentsUploaded} gradient={GRADIENTS.blue} delay={0.3} />
        <StatCard icon={Database} label="DB Records" value={apiStats?.statistics?.database_stats?.total_documents || 0} gradient={GRADIENTS.pink} delay={0.4} />
        <StatCard icon={Trophy} label="Expertise" value={`${stats.expertiseLevel}%`} gradient={GRADIENTS.gold} delay={0.5} />
      </div>

      {/* Tabs */}
      <div style={{
        background: 'white',
        borderRadius: '16px',
        padding: '1rem',
        marginBottom: '2rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
        display: 'flex',
        gap: '0.5rem',
        overflowX: 'auto'
      }}>
        {[
          { id: 'overview', label: 'Overview', icon: User },
          { id: 'activity', label: 'Activity', icon: Activity },
          { id: 'knowledge', label: 'Knowledge Base', icon: Database },
          { id: 'achievements', label: 'Achievements', icon: Trophy },
          { id: 'settings', label: 'Settings', icon: Settings }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setSelectedTab(tab.id)}
            style={{
              flex: 1,
              minWidth: 140,
              padding: '1rem 1.5rem',
              borderRadius: '12px',
              border: 'none',
              background: selectedTab === tab.id ? GRADIENTS.purple : 'transparent',
              color: selectedTab === tab.id ? 'white' : '#64748b',
              fontWeight: '700',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              transition: 'all 0.3s ease',
              boxShadow: selectedTab === tab.id ? '0 4px 12px rgba(102, 126, 234, 0.4)' : 'none'
            }}
          >
            <tab.icon size={18} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content Area */}
      {selectedTab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
          {/* About */}
          <div>
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
              marginBottom: '2rem'
            }}>
              <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                About Me
              </h3>
              {editMode ? (
                <textarea
                  value={userData.bio}
                  onChange={(e) => setUserData({...userData, bio: e.target.value})}
                  style={{
                    width: '100%',
                    minHeight: 120,
                    padding: '1rem',
                    border: '2px solid #e2e8f0',
                    borderRadius: '12px',
                    fontSize: '1rem',
                    fontFamily: 'inherit',
                    resize: 'vertical'
                  }}
                />
              ) : (
                <p style={{ margin: 0, lineHeight: 1.8, color: '#64748b', fontSize: '1rem' }}>
                  {userData.bio}
                </p>
              )}
            </div>

            {/* Recent Activity */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                  Recent Activity
                </h3>
                <div style={{
                  padding: '0.5rem 1rem',
                  background: '#667eea20',
                  color: '#667eea',
                  borderRadius: '20px',
                  fontSize: '0.85rem',
                  fontWeight: '700'
                }}>
                  {conversationHistory.filter(h => h.role === 'user').length} queries
                </div>
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {conversationHistory
                  .filter(h => h.role === 'user')
                  .slice(-5)
                  .reverse()
                  .map((item, idx) => (
                    <div key={idx} style={{
                      padding: '1rem',
                      background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
                      borderRadius: '12px',
                      borderLeft: '4px solid #667eea',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateX(8px)';
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateX(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                        <MessageSquare size={16} color="#667eea" />
                        <span style={{ fontSize: '0.85rem', color: '#94a3b8', fontWeight: '600' }}>
                          {formatTime(item.timestamp)}
                        </span>
                      </div>
                      <p style={{ margin: 0, color: '#1e293b', fontSize: '0.95rem', lineHeight: 1.6 }}>
                        {item.content?.substring(0, 120) + (item.content?.length > 120 ? '...' : '')}
                      </p>
                    </div>
                  ))}
                
                {conversationHistory.filter(h => h.role === 'user').length === 0 && (
                  <div style={{ textAlign: 'center', padding: '3rem 1rem', color: '#94a3b8' }}>
                    <MessageSquare size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p style={{ margin: 0, fontSize: '1rem' }}>No queries yet. Start exploring!</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div>
            {/* Contact Info */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
              marginBottom: '1.5rem'
            }}>
              <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                Contact Info
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: '10px',
                    background: GRADIENTS.blue,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    <Mail size={20} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Email</div>
                    {editMode ? (
                      <input
                        value={userData.email}
                        onChange={(e) => setUserData({...userData, email: e.target.value})}
                        style={{
                          width: '100%',
                          padding: '0.5rem',
                          border: '2px solid #e2e8f0',
                          borderRadius: '8px',
                          fontSize: '0.9rem'
                        }}
                      />
                    ) : (
                      <div style={{ fontSize: '0.9rem', color: '#1e293b', fontWeight: '600' }}>{userData.email}</div>
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: '10px',
                    background: GRADIENTS.green,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    <Phone size={20} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Phone</div>
                    {editMode ? (
                      <input
                        value={userData.phone}
                        onChange={(e) => setUserData({...userData, phone: e.target.value})}
                        style={{
                          width: '100%',
                          padding: '0.5rem',
                          border: '2px solid #e2e8f0',
                          borderRadius: '8px',
                          fontSize: '0.9rem'
                        }}
                      />
                    ) : (
                      <div style={{ fontSize: '0.9rem', color: '#1e293b', fontWeight: '600' }}>{userData.phone}</div>
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: '10px',
                    background: GRADIENTS.pink,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    <MapPin size={20} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Location</div>
                    <div style={{ fontSize: '0.9rem', color: '#1e293b', fontWeight: '600' }}>{userData.location}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Expertise Progress */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
              marginBottom: '1.5rem'
            }}>
              <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                Expertise Level
              </h3>
              
              <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
                <div style={{
                  width: 120,
                  height: 120,
                  borderRadius: '50%',
                  background: `conic-gradient(${GRADIENTS.gold} ${stats.expertiseLevel * 3.6}deg, #e2e8f0 0deg)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1rem',
                  position: 'relative'
                }}>
                  <div style={{
                    width: 100,
                    height: 100,
                    borderRadius: '50%',
                    background: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '2rem',
                    fontWeight: '800',
                    color: '#667eea'
                  }}>
                    {stats.expertiseLevel}%
                  </div>
                </div>
                <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#1e293b', marginBottom: '0.25rem' }}>
                  Level {Math.floor(stats.expertiseLevel / 20)}
                </div>
                <div style={{ fontSize: '0.9rem', color: '#64748b' }}>
                  {stats.expertiseLevel < 100 ? `${100 - stats.expertiseLevel}% to next level` : 'Master Level!'}
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
            }}>
              <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                System Status
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#64748b', fontSize: '0.9rem' }}>API Status</span>
                  <div style={{
                    padding: '0.25rem 0.75rem',
                    background: apiStats ? '#10b98120' : '#ef444420',
                    color: apiStats ? '#10b981' : '#ef4444',
                    borderRadius: '12px',
                    fontSize: '0.85rem',
                    fontWeight: '700'
                  }}>
                    {apiStats ? 'Connected' : 'Offline'}
                  </div>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Session Start</span>
                  <span style={{ fontWeight: '700', fontSize: '0.9rem', color: '#1e293b' }}>
                    {apiStats?.statistics?.session_info?.start_time?.split('T')[1]?.split('.')[0] || 'N/A'}
                  </span>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Database Size</span>
                  <span style={{ fontWeight: '700', fontSize: '0.9rem', color: '#1e293b' }}>
                    {apiStats?.statistics?.database_stats?.total_documents || 0} docs
                  </span>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#64748b', fontSize: '0.9rem' }}>ML Models</span>
                  <span style={{ fontWeight: '700', fontSize: '0.9rem', color: '#1e293b' }}>
                    {speciesList.length || 0} loaded
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Activity Tab */}
      {selectedTab === 'activity' && (
        <div style={{
          background: 'white',
          borderRadius: '20px',
          padding: '2rem',
          boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
              Query History
            </h3>
            <button style={{
              padding: '0.75rem 1.5rem',
              borderRadius: '12px',
              border: 'none',
              background: GRADIENTS.purple,
              color: 'white',
              fontWeight: '700',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)',
              transition: 'all 0.3s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
              <Download size={18} />
              Export History
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {conversationHistory
              .filter(h => h.role === 'user')
              .slice(-20)
              .reverse()
              .map((item, idx) => (
                <div key={idx} style={{
                  padding: '1.5rem',
                  background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                  borderRadius: '16px',
                  border: '2px solid #e2e8f0',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.1)';
                  e.currentTarget.style.borderColor = '#667eea';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                  e.currentTarget.style.borderColor = '#e2e8f0';
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <div style={{
                        width: 36,
                        height: 36,
                        borderRadius: '10px',
                        background: GRADIENTS.purple,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white'
                      }}>
                        <MessageSquare size={18} />
                      </div>
                      <div>
                        <div style={{ fontWeight: '700', color: '#1e293b', fontSize: '1rem' }}>Query #{conversationHistory.filter(h => h.role === 'user').length - idx}</div>
                        <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{formatTime(item.timestamp)}</div>
                      </div>
                    </div>
                    <div style={{
                      padding: '0.4rem 0.8rem',
                      background: '#667eea20',
                      color: '#667eea',
                      borderRadius: '12px',
                      fontSize: '0.8rem',
                      fontWeight: '700'
                    }}>
                      Text
                    </div>
                  </div>
                  <p style={{ margin: 0, color: '#64748b', fontSize: '0.95rem', lineHeight: 1.6 }}>
                    {item.content}
                  </p>
                </div>
              ))}
            
            {conversationHistory.filter(h => h.role === 'user').length === 0 && (
              <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
                <Activity size={64} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#1e293b' }}>No Activity Yet</h4>
                <p style={{ margin: 0, color: '#94a3b8' }}>Start asking questions to see your history here</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Knowledge Base Tab */}
      {selectedTab === 'knowledge' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: 48,
                height: 48,
                borderRadius: '12px',
                background: GRADIENTS.green,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}>
                <Fish size={24} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                  Fish Species
                </h3>
                <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>
                  {speciesList.length} species available
                </p>
              </div>
            </div>

            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '0.75rem',
              maxHeight: 500,
              overflowY: 'auto',
              padding: '0.5rem'
            }}>
              {speciesList.map((species, idx) => (
                <div key={idx} style={{
                  padding: '0.75rem 1rem',
                  background: 'linear-gradient(135deg, #11998e10 0%, #38ef7d10 100%)',
                  border: '2px solid #11998e30',
                  borderRadius: '12px',
                  fontSize: '0.85rem',
                  fontWeight: '600',
                  color: '#1e293b',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = GRADIENTS.green;
                  e.currentTarget.style.color = 'white';
                  e.currentTarget.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'linear-gradient(135deg, #11998e10 0%, #38ef7d10 100%)';
                  e.currentTarget.style.color = '#1e293b';
                  e.currentTarget.style.transform = 'scale(1)';
                }}>
                  {species}
                </div>
              ))}
            </div>
          </div>

          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <div style={{
                width: 48,
                height: 48,
                borderRadius: '12px',
                background: GRADIENTS.orange,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}>
                <Microscope size={24} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                  Fish Diseases
                </h3>
                <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>
                  {diseasesList.length} diseases detectable
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxHeight: 500, overflowY: 'auto' }}>
              {diseasesList.map((disease, idx) => (
                <div key={idx} style={{
                  padding: '1rem',
                  background: 'linear-gradient(135deg, #ee097910 0%, #ff6a0010 100%)',
                  border: '2px solid #ee097930',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateX(8px)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(238, 9, 121, 0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateX(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{
                      width: 32,
                      height: 32,
                      borderRadius: '8px',
                      background: GRADIENTS.orange,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white'
                    }}>
                      <Microscope size={16} />
                    </div>
                    <div style={{ fontWeight: '700', color: '#1e293b', fontSize: '0.95rem' }}>
                      {disease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Achievements Tab */}
      {selectedTab === 'achievements' && (
        <div>
          <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
            <h2 style={{ fontSize: '2rem', fontWeight: '800', color: '#1e293b', marginBottom: '0.5rem' }}>
              Your Achievements
            </h2>
            <p style={{ fontSize: '1.1rem', color: '#64748b', margin: 0 }}>
              Unlock badges by using MeenaSetu AI
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '2rem'
          }}>
            {achievements.map((achievement, index) => (
              <AchievementCard key={index} achievement={{...achievement, current: stats[achievement.field]}} index={index} />
            ))}
          </div>
        </div>
      )}

      {/* Settings Tab */}
      {selectedTab === 'settings' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
          }}>
            <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Bell size={24} color="#667eea" />
              Notifications
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {[
                { label: 'Email Notifications', desc: 'Receive updates via email', enabled: true },
                { label: 'Activity Alerts', desc: 'Get notified about new activity', enabled: true },
                { label: 'Query Reminders', desc: 'Daily query suggestions', enabled: false }
              ].map((item, idx) => (
                <div key={idx} style={{
                  padding: '1.5rem',
                  background: '#f8fafc',
                  borderRadius: '12px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div>
                    <div style={{ fontWeight: '700', color: '#1e293b', marginBottom: '0.25rem' }}>{item.label}</div>
                    <div style={{ fontSize: '0.85rem', color: '#64748b' }}>{item.desc}</div>
                  </div>
                  <button style={{
                    padding: '0.5rem 1rem',
                    borderRadius: '8px',
                    border: 'none',
                    background: item.enabled ? GRADIENTS.green : '#e2e8f0',
                    color: item.enabled ? 'white' : '#64748b',
                    fontWeight: '700',
                    fontSize: '0.85rem',
                    cursor: 'pointer'
                  }}>
                    {item.enabled ? 'Enabled' : 'Disabled'}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)'
          }}>
            <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Shield size={24} color="#667eea" />
              Security
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {[
                { label: 'Change Password', desc: 'Update your password', icon: Lock },
                { label: 'Two-Factor Auth', desc: 'Add extra security', icon: Shield },
                { label: 'API Access', desc: 'Manage API keys', icon: Settings }
              ].map((item, idx) => (
                <button key={idx} style={{
                  padding: '1.5rem',
                  background: '#f8fafc',
                  borderRadius: '12px',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  textAlign: 'left',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#667eea10';
                  e.currentTarget.style.transform = 'translateX(4px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = '#f8fafc';
                  e.currentTarget.style.transform = 'translateX(0)';
                }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: '10px',
                    background: GRADIENTS.purple,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    <item.icon size={20} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: '700', color: '#1e293b', marginBottom: '0.25rem' }}>{item.label}</div>
                    <div style={{ fontSize: '0.85rem', color: '#64748b' }}>{item.desc}</div>
                  </div>
                  <ChevronRight size={20} color="#667eea" />
                </button>
              ))}
            </div>
          </div>

          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
            gridColumn: 'span 2'
          }}>
            <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
              Data Management
            </h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
              {[
                { label: 'Export Data', desc: 'Download your profile', icon: Download, gradient: GRADIENTS.blue },
                { label: 'Clear History', desc: 'Remove query history', icon: X, gradient: GRADIENTS.orange },
                { label: 'Refresh Cache', desc: 'Reload from server', icon: RefreshCw, gradient: GRADIENTS.green }
              ].map((item, idx) => (
                <button key={idx} style={{
                  padding: '1.5rem',
                  borderRadius: '16px',
                  border: 'none',
                  background: item.gradient,
                  color: 'white',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                }}
                onClick={() => {
                  if (item.label === 'Refresh Cache') fetchAPIData();
                  else showNotification(`${item.label} clicked!`, 'success');
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                }}>
                  <item.icon size={32} style={{ marginBottom: '1rem' }} />
                  <div style={{ fontWeight: '700', fontSize: '1.1rem', marginBottom: '0.25rem' }}>{item.label}</div>
                  <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>{item.desc}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Avatar Picker Modal */}
      {showAvatarPicker && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.6)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          animation: 'fadeIn 0.3s ease'
        }}
        onClick={() => setShowAvatarPicker(false)}>
          <div style={{
            background: 'white',
            borderRadius: '24px',
            padding: '2.5rem',
            maxWidth: '500px',
            width: '90%',
            boxShadow: '0 24px 80px rgba(0,0,0,0.3)',
            animation: 'slideUp 0.3s ease'
          }}
          onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
              <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' }}>
                Choose Your Avatar
              </h3>
              <button
                onClick={() => setShowAvatarPicker(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  padding: '0.5rem',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <X size={24} color="#64748b" />
              </button>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
              {avatarOptions.map((emoji, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setUserData({...userData, avatar: emoji});
                    setShowAvatarPicker(false);
                    showNotification('Avatar updated!', 'success');
                  }}
                  style={{
                    fontSize: '3rem',
                    padding: '1.5rem',
                    borderRadius: '16px',
                    border: userData.avatar === emoji ? '3px solid #667eea' : '2px solid #e2e8f0',
                    background: userData.avatar === emoji ? '#667eea10' : 'white',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.1)';
                    e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.15)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  {emoji}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(100px); }
          to { opacity: 1; transform: translateX(0); }
        }
        ::-webkit-scrollbar {
          width: 10px;
          height: 10px;
        }
        ::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
        }
      `}</style>
    </div>
  );
};

export default EnhancedProfile;