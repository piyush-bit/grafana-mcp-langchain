import React from 'react';
import { testIds } from '../components/testIds';
import { PluginPage } from '@grafana/runtime';
import App from '../App';

function PageOne() {

  return (
    <PluginPage>
      <div data-testid={testIds.pageOne.container}>
        <App/>
      </div>
    </PluginPage>
  );
}

export default PageOne;
