"""
ASGI config for a_two_fold_machine_learning_approach.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'A_New_Data_Science_Model.settings')

application = get_asgi_application()
