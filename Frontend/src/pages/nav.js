import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav style={styles.navbar}>
      <div style={styles.logo}>
        MedAi
      </div>
      <div style={styles.navLinks}>
        <Link to="/login" style={styles.link}>
          <button style={styles.authButton}>Login</button>
        </Link>
        <Link to="/signup" style={styles.link}>
          <button style={styles.authButton}>Sign Up</button>
        </Link>
      </div>
    </nav>
  );
};

const styles = {
  navbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '15px 30px',
    backgroundColor: '#0055A4', 
    color: 'white',
    fontFamily: 'Arial, sans-serif',
  },
  logo: {
    fontSize: '24px',
    fontWeight: 'bold',
  },
  navLinks: {
    display: 'flex',
    gap: '10px',
  },
  authButton: {
    backgroundColor: '#FFFFFF', 
    color: '#0055A4',           
    border: '1px solid #0055A4',
    borderRadius: '4px',
    padding: '8px 16px',
    cursor: 'pointer',
    fontWeight: 'bold',
    transition: 'background-color 0.3s ease',
  },
  authButtonHover: {
    backgroundColor: '#0055A4',
    color: '#FFFFFF',
  },
  link: {
    textDecoration: 'none', // Removes underline from the link
  }
};

export default Navbar;
