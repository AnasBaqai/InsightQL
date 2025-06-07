<div align="center">

# 🔍 InsightQL

### _Chat with Your Database in Natural Language_

[![Next.js](https://img.shields.io/badge/Next.js-13.5.6-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![NestJS](https://img.shields.io/badge/NestJS-10.0.0-red?style=for-the-badge&logo=nestjs)](https://nestjs.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.195-green?style=for-the-badge)](https://langchain.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)](https://www.docker.com/)

_Transform the way you interact with databases using the power of AI and natural language processing_

</div>

---

## 🌟 Overview

**InsightQL** is a revolutionary database interaction platform that bridges the gap between human language and SQL databases. Using advanced AI models and LangChain SQL Agents, it enables users to query, analyze, and manipulate databases through natural language conversations.

Whether you're a data analyst, developer, or business user, InsightQL makes database operations accessible to everyone - no SQL expertise required!

## ✨ Key Features

### 🤖 **AI-Powered Query Generation**

- Convert natural language questions into optimized SQL queries
- Support for complex joins, aggregations, and nested queries
- Intelligent query suggestion and auto-completion

### 🔄 **Multi-Database Support**

- **SQLite**: Perfect for development and small applications
- **PostgreSQL**: Enterprise-grade relational database support
- **MongoDB**: NoSQL document database integration

### 🎯 **Smart Query Optimization**

- Automatic query performance analysis
- SQL dialect translation between different databases
- Query execution plan visualization and suggestions

### 📊 **Rich Data Visualization**

- Interactive table displays with sorting and filtering
- Export results in multiple formats (CSV, JSON, Excel)
- Real-time query execution monitoring

### 🔐 **Enterprise Security**

- JWT-based authentication system
- Role-based access control
- Query history and audit trails
- Secure database connection management

### 🚀 **Developer Experience**

- Docker containerized deployment
- RESTful API architecture
- Real-time WebSocket connections
- Comprehensive error handling and logging

## 🏗️ Architecture

```mermaid
graph TB
    A[Frontend - Next.js] --> B[API Gateway - NestJS]
    B --> C[LangChain SQL Agent]
    C --> D[OpenAI GPT Models]
    B --> E[(PostgreSQL)]
    B --> F[(SQLite)]
    B --> G[(MongoDB)]
    B --> H[Authentication Service]
    B --> I[Query History Service]
```

## 🛠️ Tech Stack

### **Frontend**

- **Next.js 13.5.6** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Flowbite React** - Component library
- **Material-UI** - Advanced UI components

### **Backend**

- **NestJS 10.0** - Scalable Node.js framework
- **TypeORM** - Database ORM with TypeScript support
- **Passport.js** - Authentication middleware
- **JWT** - Secure token-based authentication

### **AI & Data Processing**

- **LangChain 0.0.195** - AI agent framework
- **OpenAI API** - Large language model integration
- **SQL Agent** - Specialized database interaction agent

### **Databases**

- **PostgreSQL 12** - Primary relational database
- **SQLite 3** - Embedded database support
- **MongoDB 6.0** - Document database for logs and sessions

### **DevOps & Infrastructure**

- **Docker & Docker Compose** - Containerized deployment
- **pgAdmin** - PostgreSQL administration interface
- **ESLint & Prettier** - Code quality and formatting

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18.0 or higher
- **Docker** and **Docker Compose**
- **OpenAI API Key** ([Get yours here](https://platform.openai.com/api-keys))

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/InsightQL.git
cd InsightQL
```

### 2. Environment Setup

Create a `.env` file in the `nest_api` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Database Configuration
DB_TYPE=postgres
PG_HOST=postgres
PG_USER=postgres
PG_PASSWORD=postgres
PG_DB=postgres
PG_PORT=5432

# PostgreSQL Docker Setup
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# pgAdmin Configuration
PGADMIN_DEFAULT_EMAIL=admin@pgadmin.com
PGADMIN_DEFAULT_PASSWORD=admin

# MongoDB Configuration
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=pass12345
MONGO_URL=mongodb://root:pass12345@mongodb:27017/chatwithdb?authSource=admin

# Application Configuration
PORT=5000
JWT_SECRET=your-jwt-secret-here
NODE_ENV=development
```

### 3. Docker Deployment (Recommended)

```bash
# Navigate to the API directory
cd nest_api

# Build and start all services
docker-compose build --no-cache
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 4. Manual Installation

#### Backend Setup

```bash
cd nest_api
npm install
npm run start:dev
```

#### Frontend Setup

```bash
cd client
npm install
npm run dev
```

### 5. Access the Application

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:5000](http://localhost:5000)
- **pgAdmin**: [http://localhost:5050](http://localhost:5050)
- **API Documentation**: [http://localhost:5000/api/docs](http://localhost:5000/api/docs)

## 📖 Usage Guide

### Getting Started with Your First Query

1. **Sign Up/Login**: Create an account or sign in to access the chat interface
2. **Connect Database**: Configure your database connection in the settings
3. **Start Chatting**: Navigate to `/chat` and begin asking questions about your data

### Example Queries

```
💬 "Show me all customers from New York"
💬 "What were the top 5 products sold last month?"
💬 "Find the average order value by customer segment"
💬 "Create a summary report of sales by region"
💬 "Which employees have the highest sales performance?"
```

### Database Setup Examples

#### Loading Sample Data (PostgreSQL)

1. Access pgAdmin at `localhost:5050`
2. Login with credentials from your `.env` file
3. Create a new server connection:
   - **Host**: `postgres`
   - **Port**: `5432`
   - **Username/Password**: From `.env` file
4. Import your database dump or use the provided sample data

#### SQLite Sample Database

The application includes a sample Northwind database for testing:

```bash
cd client
npm run seed  # Loads sample data
```

## 🎯 Advanced Features

### Query History and Analytics

- All queries are stored with timestamps and performance metrics
- View and rerun previous queries
- Export query history for analysis

### Custom Database Connections

- Support for multiple database connections
- Connection pooling for optimal performance
- SSL and certificate-based authentication

### API Integration

- RESTful API endpoints for programmatic access
- Webhook support for real-time notifications
- Rate limiting and API key authentication

## 🔧 Development

### Project Structure

```
InsightQL/
├── client/                 # Next.js frontend application
│   ├── src/
│   │   ├── app/           # App router pages
│   │   ├── components/    # Reusable React components
│   │   ├── lib/          # Utility functions and configurations
│   │   └── interfaces/   # TypeScript type definitions
│   └── package.json
├── nest_api/              # NestJS backend application
│   ├── src/
│   │   ├── modules/      # Feature modules
│   │   ├── guards/       # Authentication guards
│   │   ├── dto/         # Data transfer objects
│   │   └── entities/    # Database entities
│   ├── docker-compose.yaml
│   └── package.json
└── README.md
```

### Running Tests

```bash
# Backend tests
cd nest_api
npm run test
npm run test:e2e

# Frontend tests
cd client
npm run test
```

### Code Quality

```bash
# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
```

## 🐳 Docker Configuration

The application uses Docker Compose for easy deployment with the following services:

- **nestapp**: Main NestJS API server
- **postgres**: PostgreSQL database
- **pgadmin**: Database administration interface
- **mongodb**: Document database for session storage

### Service Ports

- **Frontend**: 3000
- **Backend API**: 5000
- **PostgreSQL**: 5432
- **pgAdmin**: 5050
- **MongoDB**: 27017

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Contribution Guidelines

- Follow the existing code style and conventions
- Write comprehensive tests for new features
- Update documentation for any API changes
- Ensure all tests pass before submitting
- Use meaningful commit messages

### Areas for Contribution

- 🌐 Additional database support (MySQL, Oracle, etc.)
- 🎨 UI/UX improvements and new themes
- 📊 Advanced data visualization features
- 🔒 Enhanced security features
- 📱 Mobile responsive improvements
- 🌍 Internationalization support

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing powerful language models
- **LangChain** for the incredible SQL agent framework
- **Vercel** for Next.js and hosting solutions
- **NestJS** community for the robust backend framework
- All contributors who help make this project better

## 📞 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/InsightQL/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/InsightQL/discussions)
- 📧 **Email**: support@insightql.com
- 🐦 **Twitter**: [@InsightQL](https://twitter.com/InsightQL)

---

<div align="center">

**⭐ If you find InsightQL helpful, please consider giving it a star on GitHub! ⭐**

Made with ❤️ by the InsightQL Team

</div>
