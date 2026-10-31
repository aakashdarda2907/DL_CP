from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser


class CustomUserAdmin(UserAdmin):
    """
    Extends Django's default UserAdmin so our extra fields show up
    in the admin panel instead of being hidden.
    """
    model = CustomUser
    list_display = ['username', 'email', 'goal', 'coaching_style', 'current_weight_kg', 'target_weight_kg', 'is_staff']

    fieldsets = UserAdmin.fieldsets + (
        ('Fitness Profile', {
            'fields': (
                'height_cm', 'current_weight_kg', 'target_weight_kg',
                'age', 'gender', 'goal', 'activity_level', 'coaching_style', 'current_calorie_target',
            )
        }),
    )


admin.site.register(CustomUser, CustomUserAdmin)