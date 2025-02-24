# Use the official Node.js image
FROM node:18

# Set the working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./
# COPY .env /app/.env

# Install dependencies and run npm audit to check for vulnerabilities
RUN npm install 

# Copy the rest of your application code
COPY . .

# Expose the port your app runs on
EXPOSE 4000

# Command to run the app
CMD ["node", "index.js"]
