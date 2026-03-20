from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, URL, ValidationError

from .models import User


class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=80)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password")],
    )
    submit = SubmitField("Create account")

    def validate_username(self, field):
        if User.query.filter_by(username=field.data.strip()).first():
            raise ValidationError("That username is already in use.")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data.strip().lower()).first():
            raise ValidationError("That email is already registered.")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log in")


class AnalyzeForm(FlaskForm):
    title = StringField("Headline", validators=[Optional(), Length(max=255)])
    article_text = TextAreaField("Article Text", validators=[Optional(), Length(min=50, max=20000)])
    article_url = StringField("Article URL", validators=[Optional(), URL(), Length(max=2000)])
    submit = SubmitField("Analyze article")

    def validate(self, extra_validators=None):
        if not super().validate(extra_validators=extra_validators):
            return False

        has_text = bool((self.article_text.data or "").strip())
        has_url = bool((self.article_url.data or "").strip())
        if not has_text and not has_url:
            message = "Provide article text or a valid URL."
            self.article_text.errors.append(message)
            self.article_url.errors.append(message)
            return False
        return True
