const jwt = require('jsonwebtoken');

const ensureAuthenticated = (req, res, next) => {
    // Get the token from the Authorization header
    const auth = req.headers['authorization'];
    
    if (!auth) {
        return res.status(403).json({ message: "Unauthorized, JWT token is required" });
    }
    
  
    
    try {
        // Verify the token
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded;
        next(); // Move on to the next middleware/route handler
    } catch (err) {
        return res.status(403).json({ message: "Unauthorized, JWT token is invalid or expired" });
    }
};

module.exports = ensureAuthenticated;
