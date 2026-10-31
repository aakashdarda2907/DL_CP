from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import CustomUser


def landing_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard:home')
    return render(request, 'accounts/landing.html')


def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        # Preserve everything except passwords, so a validation error
        # doesn't force the user to retype the whole form.
        form_data = {
            'username': username,
            'email': email,
            'height_cm': request.POST.get('height_cm', ''),
            'current_weight_kg': request.POST.get('current_weight_kg', ''),
            'target_weight_kg': request.POST.get('target_weight_kg', ''),
            'age': request.POST.get('age', ''),
            'gender': request.POST.get('gender', ''),
            'goal': request.POST.get('goal', 'maintain'),
            'activity_level': request.POST.get('activity_level', 'sedentary'),
        }

        error = None
        if not username or not email or not password:
            error = 'All required fields must be filled.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters.'
        elif CustomUser.objects.filter(username=username).exists():
            error = 'Username already taken.'
        elif CustomUser.objects.filter(email=email).exists():
            error = 'Email already registered.'

        if error:
            messages.error(request, error)
            return render(request, 'accounts/signup.html', {'form_data': form_data})

        user = CustomUser.objects.create_user(username=username, email=email, password=password)
        user.height_cm = form_data['height_cm'] or None
        user.current_weight_kg = form_data['current_weight_kg'] or None
        user.target_weight_kg = form_data['target_weight_kg'] or None
        user.age = form_data['age'] or None
        user.gender = form_data['gender']
        user.goal = form_data['goal']
        user.activity_level = form_data['activity_level']
        user.save()

        login(request, user)
        messages.success(request, f'Welcome, {user.username}!')
        return redirect('dashboard:home')

    return render(request, 'accounts/signup.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard:home')
        else:
            messages.error(request, 'Invalid username or password.')
            return render(request, 'accounts/login.html', {'form_data': {'username': username}})

    return render(request, 'accounts/login.html')


def logout_view(request):
    logout(request)
    return redirect('accounts:landing')


@login_required
def profile_view(request):
    user = request.user

    if request.method == 'POST':
        user.height_cm = request.POST.get('height_cm') or user.height_cm
        user.current_weight_kg = request.POST.get('current_weight_kg') or user.current_weight_kg
        user.target_weight_kg = request.POST.get('target_weight_kg') or user.target_weight_kg
        user.age = request.POST.get('age') or user.age
        user.gender = request.POST.get('gender', user.gender)
        user.goal = request.POST.get('goal', user.goal)
        user.activity_level = request.POST.get('activity_level', user.activity_level)

        # Only accept a coaching style that actually exists (ignore tampered values)
        chosen_style = request.POST.get('coaching_style')
        if chosen_style in dict(CustomUser.COACHING_STYLE_CHOICES):
            user.coaching_style = chosen_style

        user.save()
        messages.success(request, 'Profile updated.')
        return redirect('accounts:profile')

    return render(request, 'accounts/profile.html', {'user_obj': user})
