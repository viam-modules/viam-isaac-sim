import { mount } from 'svelte';
import { useBridge } from '@viamrobotics/post-message-sdk/svelte';
import App from './App.svelte';
import './app.css';

useBridge({ reportSize: true });

mount(App, { target: document.getElementById('app')! });
