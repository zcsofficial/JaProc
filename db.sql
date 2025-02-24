CREATE DATABASE jazz;
USE jazz;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(64) NOT NULL, -- SHA-256 hash
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE exams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    suspicious_objects TEXT, -- Comma-separated list of disallowed objects
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE exam_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    exam_id INT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    warnings INT DEFAULT 0,
    status ENUM('active', 'completed', 'terminated') DEFAULT 'active',
    video_file VARCHAR(255), -- Path to recorded video
    audio_file VARCHAR(255), -- Path to recorded audio
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (exam_id) REFERENCES exams(id)
);

CREATE TABLE logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id)
);

CREATE TABLE exam_questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    exam_id INT NOT NULL,
    question_text TEXT NOT NULL,
    question_number INT NOT NULL,
    FOREIGN KEY (exam_id) REFERENCES exams(id)
);

CREATE TABLE exam_answers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question_id INT NOT NULL,
    answer_text VARCHAR(255) NOT NULL,
    is_correct BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (question_id) REFERENCES exam_questions(id)
);
CREATE TABLE user_answers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    question_id INT NOT NULL,
    answer_id INT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id),
    FOREIGN KEY (question_id) REFERENCES exam_questions(id),
    FOREIGN KEY (answer_id) REFERENCES exam_answers(id)
);
ALTER TABLE users ADD COLUMN role ENUM('user', 'admin') DEFAULT 'user';
UPDATE users SET role = 'admin' WHERE email = 'admin@example.com';
-- Sample Data
INSERT INTO users (full_name, email, password) VALUES 
('Admin User', 'admin@example.com', HASH('SHA256', 'admin123')),
('John Doe', 'john@example.com', HASH('SHA256', 'password123'));

INSERT INTO exams (name, suspicious_objects) VALUES 
('Math 101', 'cell phone,book,laptop'),
('Physics 102', 'cell phone,book,calculator');

INSERT INTO exam_questions (exam_id, question_text, question_number) VALUES 
(1, 'What is 2 + 2?', 1),
(1, 'What is the square root of 16?', 2);

INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(1, '3', FALSE),
(1, '4', TRUE),
(1, '5', FALSE),
(2, '2', FALSE),
(2, '4', TRUE),
(2, '8', FALSE);