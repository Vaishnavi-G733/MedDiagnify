# Step 1: Use the Node.js image to build the React app
FROM node:18 AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json for dependencies installation
COPY package.json package-lock.json ./ 

# Install dependencies, including nodemon if it's required
RUN npm install

# Copy all files into the container
COPY . . 

# Expose port 3000 to serve the React app
EXPOSE 3000

# Start the app using npm (React app should be started via 'npm start')
CMD ["npm", "start"]