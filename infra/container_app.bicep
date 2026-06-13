// infra/container_app.bicep
// -----------------------------------------------------------------------
// WSL Prediction Engine — Azure Container Apps Infrastructure
//
// Deploys:
//   - Container Apps Environment
//   - Container App (scale-to-zero, 0-1 replicas)
//   - Key Vault with all secrets
//   - Application Insights for monitoring
//
// Deploy with:
//   az deployment group create \
//     --resource-group wsl-analytics-rg \
//     --template-file infra/container_app.bicep \
//     --parameters @infra/params.json
// -----------------------------------------------------------------------

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Container image to deploy (from ACR)')
param containerImage string

@description('Supabase project URL')
@secure()
param supabaseUrl string

@description('Supabase service role key')
@secure()
param supabaseKey string

@description('API key for prediction endpoints')
@secure()
param apiKey string

// ---------------------------------------------------------------------------
// Log Analytics (required by Container Apps)
// ---------------------------------------------------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'wsl-analytics-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ---------------------------------------------------------------------------
// Application Insights
// ---------------------------------------------------------------------------

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'wsl-prediction-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ---------------------------------------------------------------------------
// Key Vault
// ---------------------------------------------------------------------------

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'wsl-prediction-kv'
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    softDeleteRetentionInDays: 7
  }
}

resource kvSupabaseUrl 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'supabase-url'
  properties: { value: supabaseUrl }
}

resource kvSupabaseKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'supabase-key'
  properties: { value: supabaseKey }
}

resource kvApiKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'api-key'
  properties: { value: apiKey }
}

resource kvAppInsights 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'appinsights-connection-string'
  properties: { value: appInsights.properties.ConnectionString }
}

// ---------------------------------------------------------------------------
// Container Apps Environment
// ---------------------------------------------------------------------------

resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'wsl-analytics-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Container App
// ---------------------------------------------------------------------------

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'wsl-prediction-engine'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
      secrets: [
        { name: 'supabase-url', keyVaultUrl: kvSupabaseUrl.properties.secretUri, identity: 'system' }
        { name: 'supabase-key', keyVaultUrl: kvSupabaseKey.properties.secretUri, identity: 'system' }
        { name: 'api-key', keyVaultUrl: kvApiKey.properties.secretUri, identity: 'system' }
        { name: 'appinsights-connection-string', keyVaultUrl: kvAppInsights.properties.secretUri, identity: 'system' }
      ]
    }
    template: {
      containers: [
        {
          name: 'wsl-prediction-engine'
          image: containerImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'SUPABASE_URL', secretRef: 'supabase-url' }
            { name: 'SUPABASE_SERVICE_ROLE_KEY', secretRef: 'supabase-key' }
            { name: 'API_KEY', secretRef: 'api-key' }
            { name: 'APPINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-connection-string' }
          ]
        }
      ]
      // Scale to zero — costs nothing when idle
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '10' } }
          }
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output appInsightsConnectionString string = appInsights.properties.ConnectionString
