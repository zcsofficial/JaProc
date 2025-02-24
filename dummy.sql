-- Insert the new exam
INSERT INTO exams (name, suspicious_objects) 
VALUES ('Basic Computer Skills', 'cell phone,book,notes');

-- Get the exam_id (assuming it’s the last inserted exam, adjust if necessary)
SET @exam_id = LAST_INSERT_ID();

-- Insert Questions
INSERT INTO exam_questions (exam_id, question_text, question_number) VALUES 
(@exam_id, 'What is the primary function of the CPU?', 1),
(@exam_id, 'Which of these is an input device?', 2),
(@exam_id, 'What does RAM stand for?', 3),
(@exam_id, 'Which file extension is commonly used for Microsoft Word documents?', 4),
(@exam_id, 'What is the purpose of an operating system?', 5),
(@exam_id, 'Which of these is a web browser?', 6),
(@exam_id, 'What is the main storage device in a computer?', 7),
(@exam_id, 'Which key is used to copy text or files?', 8),
(@exam_id, 'What does USB stand for?', 9),
(@exam_id, 'Which of these is a type of software?', 10);

-- Insert Answers for Question 1
SET @q1_id = LAST_INSERT_ID();
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q1_id, 'To store data permanently', FALSE),
(@q1_id, 'To process instructions from programs', TRUE),
(@q1_id, 'To display graphics', FALSE),
(@q1_id, 'To connect to the internet', FALSE);

-- Insert Answers for Question 2
SET @q2_id = @q1_id + 1;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q2_id, 'Monitor', FALSE),
(@q2_id, 'Keyboard', TRUE),
(@q2_id, 'Printer', FALSE),
(@q2_id, 'Speaker', FALSE);

-- Insert Answers for Question 3
SET @q3_id = @q1_id + 2;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q3_id, 'Random Access Memory', TRUE),
(@q3_id, 'Read Always Memory', FALSE),
(@q3_id, 'Rapid Access Module', FALSE),
(@q3_id, 'Remote Access Memory', FALSE);

-- Insert Answers for Question 4
SET @q4_id = @q1_id + 3;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q4_id, '.xlsx', FALSE),
(@q4_id, '.docx', TRUE),
(@q4_id, '.pptx', FALSE),
(@q4_id, '.txt', FALSE);

-- Insert Answers for Question 5
SET @q5_id = @q1_id + 4;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q5_id, 'To manage hardware and software resources', TRUE),
(@q5_id, 'To create documents', FALSE),
(@q5_id, 'To browse the internet', FALSE),
(@q5_id, 'To store files permanently', FALSE);

-- Insert Answers for Question 6
SET @q6_id = @q1_id + 5;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q6_id, 'Microsoft Excel', FALSE),
(@q6_id, 'Google Chrome', TRUE),
(@q6_id, 'Adobe Photoshop', FALSE),
(@q6_id, 'Notepad', FALSE);

-- Insert Answers for Question 7
SET @q7_id = @q1_id + 6;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q7_id, 'Hard Disk Drive (HDD)', TRUE),
(@q7_id, 'Random Access Memory (RAM)', FALSE),
(@q7_id, 'Central Processing Unit (CPU)', FALSE),
(@q7_id, 'Graphics Processing Unit (GPU)', FALSE);

-- Insert Answers for Question 8
SET @q8_id = @q1_id + 7;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q8_id, 'Ctrl + V', FALSE),
(@q8_id, 'Ctrl + C', TRUE),
(@q8_id, 'Ctrl + X', FALSE),
(@q8_id, 'Ctrl + P', FALSE);

-- Insert Answers for Question 9
SET @q9_id = @q1_id + 8;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q9_id, 'Universal Serial Bus', TRUE),
(@q9_id, 'Ultra Speed Bandwidth', FALSE),
(@q9_id, 'User System Backup', FALSE),
(@q9_id, 'Unified Storage Base', FALSE);

-- Insert Answers for Question 10
SET @q10_id = @q1_id + 9;
INSERT INTO exam_answers (question_id, answer_text, is_correct) VALUES 
(@q10_id, 'Monitor', FALSE),
(@q10_id, 'Operating System', TRUE),
(@q10_id, 'Keyboard', FALSE),
(@q10_id, 'Hard Drive', FALSE);