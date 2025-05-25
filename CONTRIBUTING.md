# Contributing to Multi-Face Attendance System

Thank you for your interest in contributing to the Multi-Face Attendance System! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### 1. Fork the Repository
- Fork the project on GitHub
- Clone your fork locally
- Create a new branch for your feature or bug fix

### 2. Set Up Development Environment
```bash
# Clone your fork
git clone https://github.com/iamshuklau/multi-face-attendance-system.git
cd multi-face-attendance-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 3. Make Your Changes
- Write clean, readable code
- Follow the existing code style
- Add comments and docstrings
- Include type hints where appropriate

### 4. Test Your Changes
```bash
# Run the test suite
python test_system.py

# Test the application manually
streamlit run app_comprehensive.py
```

### 5. Submit a Pull Request
- Push your changes to your fork
- Create a pull request with a clear description
- Reference any related issues

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Keep functions small and focused
- Add docstrings to all functions and classes

### Example Function Documentation
```python
def process_attendance(student_data: List[Dict], confidence_threshold: float = 0.6) -> Dict:
    """
    Process attendance data for a group of students.
    
    Args:
        student_data: List of student information dictionaries
        confidence_threshold: Minimum confidence for face recognition
        
    Returns:
        Dictionary containing processed attendance results
        
    Raises:
        ValueError: If student_data is empty or invalid
    """
    pass
```

### Commit Message Format
```
type(scope): brief description

Detailed explanation of the changes made.

- List specific changes
- Reference issues if applicable

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Screenshots if applicable

### Feature Requests
For new features, please provide:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach
- Any relevant examples or mockups

## 🔧 Development Guidelines

### Adding New Features
1. Check if the feature aligns with project goals
2. Create an issue to discuss the feature first
3. Implement the feature with proper tests
4. Update documentation as needed
5. Ensure backward compatibility

### Code Review Process
- All changes require review before merging
- Address reviewer feedback promptly
- Maintain a respectful and constructive tone
- Be open to suggestions and improvements

### Testing Requirements
- Write unit tests for new functions
- Ensure existing tests still pass
- Test edge cases and error conditions
- Verify performance impact

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions
- Include type hints for function parameters
- Document complex algorithms and logic
- Keep comments up to date with code changes

### README Updates
- Update installation instructions if needed
- Add new features to the features list
- Update screenshots if UI changes
- Keep the changelog current

## 🚀 Release Process

### Version Numbering
We follow semantic versioning (SemVer):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version number is bumped
- [ ] Changelog is updated
- [ ] Release notes are prepared

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations
- Bug fixes and stability improvements
- Documentation improvements
- Test coverage expansion

### Medium Priority
- New face recognition algorithms
- Additional export formats
- UI/UX improvements
- Mobile responsiveness

### Low Priority
- Code refactoring
- Additional language support
- Integration with external systems
- Advanced analytics features

## 📞 Getting Help

### Communication Channels
- GitHub Issues: For bugs and feature requests
- GitHub Discussions: For questions and general discussion
- Email: For private inquiries

### Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [face_recognition Library](https://github.com/ageitgey/face_recognition)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Multi-Face Attendance System! 🎓 